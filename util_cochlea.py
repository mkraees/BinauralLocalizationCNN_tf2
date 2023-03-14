import os
import sys
import pdb
import numpy as np
import tensorflow as tf

import util_signal

ROOT_MOUNT_POINT = os.environ.get('ROOT_MOUNT_POINT', '')


def cochlea(tensor_input,
            sr_input=20e3,
            sr_cochlea=None,
            sr_output=None,
            dtype=tf.float32,
            config_filterbank={},
            config_subband_processing={},
            kwargs_fir_lowpass_filter_input={},
            kwargs_fir_lowpass_filter_output={},
            kwargs_custom_slice={},
            kwargs_sigmoid_rate_level_function={},
            kwargs_spike_rate_level_function={},
            kwargs_spike_rate_noise={},
            kwargs_spike_generator_binomial={}):
    """
    """
    # Initialize sequential keras model for cochlear processing stages 
    model = tf.keras.Sequential()
    
    # Resample audio from sr_input to sr_cochlea
    if sr_cochlea is None:
        sr_cochlea = sr_input
    if sr_output is None:
        sr_output = sr_input
    if (not sr_input == sr_cochlea) or (kwargs_fir_lowpass_filter_input):
        lambda_resample = lambda x: util_signal.tf_fir_resample(
            x,
            sr_input=sr_input,
            sr_output=sr_cochlea,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_input,
            verbose=False)
        model.add(tf.keras.layers.Lambda(lambda_resample, name='resample_input'))
    
    # Convert audio to subbands as specified by config_filterbank
    filterbank_mode = config_filterbank.pop('mode', None)
    print('[cochlea] converting audio to subbands using {}'.format(filterbank_mode))
    if filterbank_mode is None:
        assert not config_filterbank, "filterbank_mode must be specified in config_filterbank"
        assert len(tensor_input.shape) >= 3, "shape must be [batch, time, freq, (channel)] to skip filterbank"
    elif filterbank_mode == 'connear':
        _, connear_model = connear(
            model(tensor_input),
            sr_cochlea,
            **config_filterbank)
        connear_model._name = 'filterbank_connear'
        model.add(connear_model)
    elif filterbank_mode == 'fir_gammatone_filterbank':
        lambda_filterbank = lambda x: util_signal.fir_gammatone_filterbank(
            x,
            sr_cochlea,
            **config_filterbank)[0]
        model.add(tf.keras.layers.Lambda(lambda_filterbank, dtype=dtype, name='filterbank'))
    elif filterbank_mode == 'rnn_gammatone_filterbank':
        lambda_filterbank = lambda x: util_signal.rnn_gammatone_filterbank(
            x,
            sr_cochlea,
            **config_filterbank)[0]
        model.add(tf.keras.layers.Lambda(lambda_filterbank, dtype=dtype, name='filterbank'))
    elif filterbank_mode == 'roex_filterbank':
        lambda_filterbank = lambda x: util_signal.roex_filterbank(
            x,
            sr_cochlea,
            **config_filterbank)[0]
        model.add(tf.keras.layers.Lambda(lambda_filterbank, dtype=dtype, name='filterbank'))
    elif filterbank_mode == 'half_cosine_filterbank':
        lambda_filterbank = lambda x: util_signal.half_cosine_filterbank(
            x,
            sr_cochlea,
            **config_filterbank)[0]
        model.add(tf.keras.layers.Lambda(lambda_filterbank, dtype=dtype, name='filterbank'))
    else:
        raise NotImplementedError("filterbank_mode={} not recognized".format(filterbank_mode))
    
    # Half-wave rectify subbands
    if config_subband_processing.get('rectify', False):
        model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_subbands'))
        print('[cochlea] half-wave rectified subbands')
    # Resample subbands from sr_cochlea to sr_output
    if (not sr_output == sr_cochlea) or (kwargs_fir_lowpass_filter_output):
        lambda_resample = lambda x: util_signal.tf_fir_resample(
            x,
            sr_input=sr_cochlea,
            sr_output=sr_output,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_output,
            verbose=False)
        model.add(tf.keras.layers.Lambda(lambda_resample, dtype=dtype, name='resample_subbands'))
        print('[cochlea] resampled subbands from {} Hz to {} Hz with filter: {}'.format(
            int(sr_cochlea), int(sr_output), kwargs_fir_lowpass_filter_output))
        # Half-wave rectify subbands again after resampling
        if config_subband_processing.get('rectify', False):
            model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_resampled_subbands'))
            print('[cochlea] half-wave rectified resampled subbands')
    
    # Apply power compression
    power_compression = config_subband_processing.get('power_compression', None)
    if power_compression is not None:
        
        @tf.custom_gradient
        def stable_power_compression(x):
            """
            This function wraps around tf.math.pow and adds a custom gradient
            for stable back-propagation. Gradients are clipped to [-1, 1] and
            nan values are removed. Function is modeled after tfcochleagram.
            """
            y = tf.math.pow(x, power_compression)
            def grad(dy):
                g = power_compression * tf.math.pow(x, power_compression - 1)
                nan_mask = tf.math.is_nan(g)
                nan_replace = tf.ones_like(g)
                return dy * tf.where(nan_mask, nan_replace, tf.clip_by_value(g, -1, 1))
            return y, grad
        
        model.add(tf.keras.layers.Lambda(stable_power_compression, dtype=dtype, name='stable_power_compression'))
        print('[cochlea] applied {} power compression to subbands'.format(power_compression))
    
    # Slice cochlear representation (useful for selecting channels / spont rates)
    if kwargs_custom_slice:
        lambda_custom_slice = tf.keras.layers.Lambda(
            lambda x: custom_slice(x, **kwargs_custom_slice),
            dtype=dtype,
            name='custom_slice')
        model.add(lambda_custom_slice)
        print('[cochlea] incorporated custom_slice: {}'.format(kwargs_custom_slice))
    
    # Convert from amplitude / sound level units to ANF spike rates
    msg = "SPECIFY kwargs_sigmoid_rate_level_function OR kwargs_spike_rate_level_function, NOT BOTH"
    if kwargs_sigmoid_rate_level_function:
        assert not kwargs_spike_rate_level_function, msg
        lambda_sigmoid_rate_level_function = tf.keras.layers.Lambda(
            lambda x: sigmoid_rate_level_function(x, **kwargs_sigmoid_rate_level_function),
            dtype=dtype,
            name='sigmoid_rate_level_function')
        model.add(lambda_sigmoid_rate_level_function)
        print('[cochlea] incorporated sigmoid_rate_level_function: {}'.format(kwargs_sigmoid_rate_level_function))
    if kwargs_spike_rate_level_function:
        assert not kwargs_sigmoid_rate_level_function, msg
        lambda_spike_rate_level_function = tf.keras.layers.Lambda(
            lambda x: spike_rate_level_function(x, **kwargs_spike_rate_level_function),
            dtype=dtype,
            name='spike_rate_level_function')
        model.add(lambda_spike_rate_level_function)
        print('[cochlea] incorporated spike_rate_level_function: {}'.format(kwargs_spike_rate_level_function))
    
    # Add noise to spike rates
    if kwargs_spike_rate_noise:
        lambda_spike_rate_noise = tf.keras.layers.Lambda(
            lambda x: spike_rate_noise(x, **kwargs_spike_rate_noise),
            dtype=dtype,
            name='spike_rate_noise')
        model.add(lambda_spike_rate_noise)
        print('[cochlea] incorporated spike_rate_noise: {}'.format(kwargs_spike_rate_noise))
    
    # Convert spike rates to stochastic spike counts with binomial spike generator
    if kwargs_spike_generator_binomial:
        if 'sr' not in kwargs_spike_generator_binomial:
            kwargs_spike_generator_binomial['sr'] = sr_output
            print('[cochlea] inferring `sr={}` for spike_generator_binomial'.format(sr_output))
        lambda_spike_generator_binomial = tf.keras.layers.Lambda(
            lambda x: spike_generator_binomial(x, **kwargs_spike_generator_binomial),
            dtype=dtype,
            name='spike_generator_binomial')
        model.add(lambda_spike_generator_binomial)
        print('[cochlea] incorporated spike_generator_binomial: {}'.format(kwargs_spike_generator_binomial))
    
    tensor_output = None
    if tensor_input is not None:
        tensor_output = model(tensor_input)
    return tensor_output, model


def connear(tensor_input,
            sr_input=20e3,
            sr_strict=True,
            fn_connear_model='/python-packages/CoNNear_cochlea/connear/Gmodel.json',
            fn_connear_weights='/python-packages/CoNNear_cochlea/connear/Gmodel.h5',
            fn_connear_cfs='/python-packages/CoNNear_cochlea/tlmodel/cf.txt',
            cfs_spec=[1,-1,2],
            pad_crop=256,
            frame_size=16,
            trainable=False,
            load_weights=True):
    """
    Wrapper function for interacting with CoNNear peripheral auditory model
    (https://github.com/HearingTechnology/CoNNear_cochlea).
    
    Args
    ----
    tensor_input (tensor): input audio tensor with shape [batch, time]
    sr_input (int): audio sampling rate in Hz (CoNNear expects 20000 Hz)
    sr_strict (bool): if True, a value error will be raised if sr_input is not 20000 Hz
    fn_connear_model (str): JSON filename specifying CoNNear keras model architecture
    fn_connear_weights (str): filename specifying CoNNear keras model pre-trained weights
    fn_connear_cfs (str): filename specifying list of CFs for pre-trained CoNNear model
    cfs_spec (np.array): specifies which CFs to keep in output representation
    pad_crop (int): number of silent samples (20000 Hz sampling rate) added to each end
        of audio to offset CoNNear context cropping
    frame_size (int): CoNNear frame size (audio passed into CoNNear graph is silent-
        padded to have length equal to an integer multiple of frame_size)
    trainable (bool): If True, CoNNear model weights are trainable
    load_weights (bool): if True, CoNNear model weights are loaded from fn_connear_weights
    
    Returns
    -------
    tensor_output (tensor): output tensor of CoNNear with shape [batch, freq, time]
    connear_model (keras.Model): CoNNNear model object
    """
    # Check if audio sampling rate matches expected 20 kHz
    if not sr_input == 20e3:
        if sr_strict:
            raise ValueError("CoNNear operates on audio with sampling rate 20000 Hz")
        else:
            print("[connear] operating CoNNear on {} Hz audio".format(sr_input))
    
    # Initialize sequential keras model for CoNNear pre/post processing stages 
    connear_model = tf.keras.Sequential()
    
    # Reshape audio input to fit CoNNear graph [batch, time, 1]
    batch_size = tensor_input.shape[0]
    input_shape = tensor_input.shape[1:]
    layer_reshape = tf.keras.layers.Reshape(
        (-1, 1),
        input_shape=input_shape,
        batch_size=batch_size,
        name='reshape_audio_to_batch_time_1')
    connear_model.add(layer_reshape)
    
    # Pad input time dimension to offset CoNNear context cropping and fit frame size
    N = int(np.ceil((tensor_input.shape[1] + (2 * pad_crop)) / frame_size) * frame_size)
    pad_frame = N - tensor_input.shape[1] - (2 * pad_crop)
    layer_pad = tf.keras.layers.ZeroPadding1D(
        padding=(pad_crop, pad_crop + pad_frame),
        name='zero_padding_in_time')
    connear_model.add(layer_pad)
    
    # Load CoNNear graph as keras model from JSON and insert as layer
    with open(ROOT_MOUNT_POINT + fn_connear_model, 'r') as f_connear_model:
        layer_connear = tf.keras.models.model_from_json(f_connear_model.read())
        layer_connear._name = 'connear_model_from_json'
    layer_connear.trainable = trainable
    if load_weights:
        layer_connear.load_weights(ROOT_MOUNT_POINT + fn_connear_weights)
        print('[connear] loaded CoNNear weights from: {}'.format(fn_connear_weights))
    connear_model.add(layer_connear)
    
    # Rearrange CoNNear output dimensions to [batch, cf, time]
    layer_transpose = tf.keras.layers.Permute(dims=[2, 1], name='transpose_to_batch_cf_time')
    connear_model.add(layer_transpose)
    
    # Load list of CoNNear CFs; slice and re-order as specified
    cfs = np.loadtxt(ROOT_MOUNT_POINT + fn_connear_cfs) * 1e3
    if cfs_spec is not None:
        cfs_spec = np.array(cfs_spec)
        assert len(cfs_spec.shape) == 1, "cfs_spec must be a 1D array"
        if cfs_spec.dtype == int:
            cfs_mode = 'slice (start, stop, step) applied to CoNNear CFs'
            cfs_mask = np.zeros_like(cfs).astype(np.bool)
            cfs_mask[slice(*cfs_spec)] = True
        else:
            cfs_mode = 'list of accepted CoNNear CFs'
            cfs_mask = [np.isclose(np.abs(cf-cfs_spec).min(), 0) for cf in cfs]
            cfs_mask = np.array(cfs_mask, dtype=np.bool)
        print('[connear] interpreting `cfs_spec` as {} ({} of {} CFs)'.format(
            cfs_mode, cfs_mask.sum(), cfs_mask.shape[0]))
        cfs = cfs[cfs_mask]
        layer_slice = tf.keras.layers.Lambda(
            lambda x: tf.boolean_mask(x, cfs_mask, axis=1),
            name='slice_cfs')
        connear_model.add(layer_slice)
    if (len(cfs) > 1) and (cfs[0] > cfs[1]):
        print('[connear] re-ordering CF list to be in ascending order')
        cfs = np.flip(cfs)
        layer_reverse = tf.keras.layers.Lambda(
            lambda x: tf.reverse(x, axis=[1]),
            name='reverse_cfs')
        connear_model.add(layer_reverse)
    # Set static shape of CoNNear output and truncate silent padding
    layer_static_shape = tf.keras.layers.Reshape(
        (cfs.shape[0], input_shape[0] + pad_frame),
        name='set_static_shape')
    connear_model.add(layer_static_shape)
    if pad_frame > 0:
        layer_pad_remove = tf.keras.layers.Lambda(
            lambda x: x[:, :, :-pad_frame],
            name='remove_zero_padding_in_time')
        connear_model.add(layer_pad_remove)
    return connear_model(tensor_input), connear_model


def subsample_and_reduce_sum(tensor_input, axis=1, n_subsample=20, keepdims=False):
    """
    """
    tensor_output = tensor_input
    if n_subsample is not None:
        msg = "n_subsample must be <= to {}".format(tensor_input.shape[axis])
        assert n_subsample <= tensor_input.shape[axis], msg
        axis_indices = tf.range(tensor_input.shape[axis])
        subsampled_indices = tf.random.shuffle(axis_indices)[:n_subsample]
        tensor_output = tf.gather(tensor_output, subsampled_indices, axis=axis)
    tensor_output = tf.math.reduce_sum(tensor_output, axis=axis, keepdims=keepdims)
    return tensor_output


def custom_slice(x, axis=-1, args=None, expand_dims=True):
    """
    """
    while expand_dims and (axis >= len(x.shape)):
        x = tf.expand_dims(x, axis=-1)
    idx = [slice(None) for _ in range(len(x.shape))]
    if (axis < len(x.shape)) and (axis >= -len(x.shape)):
        if args is not None:
            idx[axis] = slice(*args)
    else:
        msg = "WARNING: custom_slice `axis={}` incompatible with `shape={}`"
        print(msg.format(axis, x.shape))
    return x[tuple(idx)]


def random_slice(tensor_input, slice_length, axis=-1, buffer=None):
    """
    """
    if isinstance(buffer, int):
        buffer_start = buffer
        buffer_end = buffer
    elif isinstance(buffer, tuple):
        buffer_start, buffer_end = buffer
    else:
        buffer_start = 0
        buffer_end = 0
    idx = tf.random.uniform(
        shape=(),
        minval=buffer_start,
        maxval=tensor_input.shape[axis] + 1 - buffer_end - slice_length,
        dtype=tf.int64)
    slice_index = [slice(None, None, None) for _ in range(len(tensor_input.shape))]
    slice_index[axis] = slice(idx, idx + slice_length, None)
    return tensor_input[slice_index]


def sigmoid_rate_level_function(tensor_subbands,
                                rate_spont=70.0,
                                rate_max=250.0,
                                threshold=0.0,
                                dynamic_range=25.0,
                                dynamic_range_interval=0.95):
    """
    Function implements a generalized sigmoid function as a simple
    auditory nerve rate-level function. Arguments may be specified as
    single values (same rate-level function applied to all subbands)
    or lists (different rate-level function to applied to each channel).
    Currently, different rate-level functions can only be specified for
    channel dimension, NOT frequency dimension.
    
    Args
    ----
    tensor_subbands (tensor): shape must be [batch, freq, time, (channel)] (units Pa)
    rate_spont (float or list): spontaneous firing rate (spikes/s)
    rate_max (float or list): maximum firing rate (spikes/s)
    threshold (float or list): dB SPL at which firing rate begins to increase
    dynamic range (float or list): range in dB over which firing rate changes
    dynamic_range_interval (float): fraction between 0 and 1 determines range of
        normalized firing rates to consider as the dynamic range (0.95 --> dynamic
        range spans 95% of spike rate variation and threshold refers to 2.5% increase
        in normalized spiking)
    
    Returns
    -------
    tensor_rates (tensor): spiking rate tensor with shape matched to tensor_subbands,
        possibly with expanded channel dimension (units spikes/s or normalized)
    """
    rate_spont = np.array(rate_spont).reshape([-1])
    rate_max = np.array(rate_max).reshape([-1])
    threshold = np.array(threshold).reshape([-1])
    dynamic_range = np.array(dynamic_range).reshape([-1])
    # Check arguments and see lengths are consistent (single values or channel-specific)
    y_threshold = (1 - dynamic_range_interval) / 2
    k = np.log((1 / y_threshold) - 1) / (dynamic_range / 2)
    x0 = threshold - (np.log((1 / y_threshold) - 1) / (-k))
    assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
    argument_lengths = [len(rate_spont), len(rate_max), len(threshold), len(dynamic_range)]
    n_channels = max(argument_lengths)
    msg = "inconsistent argument lengths for rate_spont, rate_max, threshold, dynamic_range"
    assert np.all([_ in (1, n_channels) for _ in argument_lengths]), msg
    channel_specific_shape = [1, 1, 1, n_channels]
    if len(tensor_subbands.shape) == 4:
        msg = "number of channels in tensor_subbands must be 1 or {}".format(n_channels)
        assert tensor_subbands.shape[-1] in (1, n_channels), msg
    if len(tensor_subbands.shape) == 3:
        if n_channels == 1:
            # Do not create new axis if n_channels == 1
            channel_specific_shape = channel_specific_shape[:-1]
        else:
            # Add channel axis if channel-specific arguments are specified
            tensor_subbands = tf.expand_dims(tensor_subbands, -1)
    msg = "tensor_subbands must have shape [batch, freq, time, (channel)]"
    assert len(tensor_subbands.shape) in (3, 4), msg
    # Convert arguments to tensors (can be single values or channel-specific)
    rate_spont = tf.constant(rate_spont, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    rate_max = tf.constant(rate_max, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    k = tf.constant(k, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    x0 = tf.constant(x0, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    # Compute sigmoid function in tensorflow with broadcasting
    x = 20.0 * tf.math.log(tensor_subbands / 20e-6) / tf.cast(tf.math.log(10.0), tensor_subbands.dtype)
    y = 1.0 / (1.0 + tf.math.exp(-k * (x - x0)))
    tensor_rates = rate_spont + (rate_max - rate_spont) * y
    return tensor_rates


def spike_rate_level_function(tensor_subbands,
                              P_0=0.5,
                              beta=3.0,
                              rate_spont=75.0,
                              rate_max=150.0,
                              rate_normalize=False):
    """
    Function implements the auditory nerve rate-level function described by Peter Heil
    and colleagues (2011, J. Neurosci.): the "amplitude-additivity model". Arguments
    may be specified as single values (same rate-level function applied to all
    subbands) or lists (different rate-level function to applied to each channel).
    Currently, different rate-level functions can only be specified for channel dimension,
    NOT frequency dimension.
    
    Args
    ----
    tensor_subbands (tensor): shape must be [batch, freq, time, (channel)] (units Pa)
    P_0 (float or list): sets operating point of rate-level function (units Pa)
    beta (float or list): determines the steepness of rate-level function (dimensionless)
    rate_spont (float or list): spontaneous firing rate (spikes/s) -- determines intrinsic sensitivity of ANF
    rate_max (float or list): maximum firing rate (spikes/s) -- determines intrinsic sensitivity of ANF
    rate_normalize (bool): if True, output will be re-scaled between 0 and 1
    
    Returns
    -------
    tensor_rates (tensor): spiking rate tensor with shape matched to tensor_subbands,
        possibly with expanded channel dimension (units spikes/s or normalized)
    """
    # Check arguments and see lengths are consistent (single values or channel-specific)
    rate_spont = np.array(rate_spont).reshape([-1])
    rate_max = np.array(rate_max).reshape([-1])
    beta = np.array(beta).reshape([-1])
    P_0 = np.array(P_0).reshape([-1])
    assert np.all(rate_spont > 0), "rate_spont must be greater than zero to avoid division by zero"
    assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
    argument_lengths = [len(rate_spont), len(rate_max), len(beta > 1), len(P_0 > 1)]
    n_channels = max(argument_lengths)
    msg = "Inconsistent argument lengths for rate_spont, rate_max, beta, P_0"
    assert np.all([_ in (1, n_channels) for _ in argument_lengths]), msg
    channel_specific_shape = [1, 1, 1, n_channels]
    if len(tensor_subbands.shape) == 4:
        assert tensor_subbands.shape[-1] in (1, n_channels)
    if len(tensor_subbands.shape) == 3:
        if n_channels == 1:
            # Do not create new axis if n_channels == 1
            channel_specific_shape = channel_specific_shape[:-1]
        else:
            # Add channel axis if channel-specific arguments are specified
            tensor_subbands = tf.expand_dims(tensor_subbands, -1)
    msg = "tensor_subbands must have shape [batch, freq, time, (channel)]"
    assert len(tensor_subbands.shape) in (3, 4), msg
    # Convert arguments to tensors (can be single values or channel-specific)
    beta = tf.constant(beta, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    P_0 = tf.constant(P_0, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    R_spont = tf.constant(rate_spont, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    R_max = tf.constant(rate_max, dtype=tensor_subbands.dtype, shape=channel_specific_shape)
    # Rate-level function implemented as Equation [8] from Heil et al. (2011, J. Neurosci.)
    S = R_spont / (R_max - R_spont) # S is "intrinsic sensitivity" of ANF (~1 for HSR fiber)
    R_func = lambda P: R_max / (1 + (tf.math.pow(S, -1) * tf.math.pow(P / P_0 + 1, -beta)))
    # Apply rate-level function to the subbands
    tensor_rates = R_func(tensor_subbands)
    # If rate_normalize is True, re-scale firing rates to fall between 0 and 1
    if rate_normalize:
        tensor_rates = (tensor_rates - R_spont) / (R_max - R_spont)
    return tensor_rates


def spike_rate_noise(tensor_rates,
                     gaussian_noise_mean=0.0,
                     gaussian_noise_stddev=1.0,
                     post_rectify=True):
    """
    Functions applies additive Gaussian noise to spike rate tensor.
    
    Args
    ----
    tensor_rates (tensor): shape must be [batch, freq, time, (channel)] (units spikes/s)
    gaussian_noise_mean (float): mean of additive Gaussian noise
    gaussian_noise_stddev (float): standard deviation of additive Gaussian noise
    post_rectify (bool): if True, noisy output is passed through ReLU
    
    Returns
    -------
    tensor_rates (dict): spike rates with additive Gaussian noise
    """
    noise = tf.random.normal(
        tf.shape(tensor_rates),
        mean=gaussian_noise_mean,
        stddev=gaussian_noise_stddev,
        dtype=tensor_rates.dtype)
    tensor_rates = tensor_rates + noise
    if post_rectify:
        tensor_rates = tf.nn.relu(tensor_rates)
    return tensor_rates


@tf.function
def map_fn_binomial(args):
    """
    Parallelizable binomial sampling function that
    explicitly sums `n` Bernoulli random variables.
    """
    (n, n_per_step, p, p_noise_stddev) = args
    counts = tf.zeros_like(p)
    if n_per_step > 0:
        sample_shape = tf.concat([[n_per_step], tf.shape(p)], axis=0)
        for _ in range(n // n_per_step):
            if p_noise_stddev > 0:
                p_noise = tf.random.normal(
                    sample_shape,
                    mean=0.0,
                    stddev=tf.cast(p_noise_stddev, p.dtype),
                    dtype=p.dtype)
                sample = tf.nn.relu(
                    tf.sign(
                        p + p_noise - tf.random.uniform(sample_shape, dtype=p.dtype)
                    )
                )
            else:
                sample = tf.nn.relu(tf.sign(p - tf.random.uniform(sample_shape, dtype=p.dtype)))
            counts = counts + tf.reduce_sum(sample, axis=0)
    return counts


@tf.function
def spike_generator_binomial(tensor_rates,
                             sr,
                             n_per_channel=1,
                             n_per_step=None,
                             p_dtype='float32',
                             p_noise_stddev=None):
    """
    Fast binomial spike generator. Spike counts are sampled independently at
    each time-frequency bin using tensor_rates * sr as p and 
    `n_per_channel` as n. Binomial sampling is performed by explicitly summing
    Bernoulli random variables.
    
    Args
    ----
    tensor_rates (tensor): shape must be [batch, freq, time, (channel)] (units spikes/s)
    sr (int): sampling rate of tensor_rates (Hz)
    n_per_channel (int or list): number of spike trains to generate per channel
        # TODO: channel refers only to last dimension of `tensor_rates`...
        # should be generalized to other dimensions (e.g., freq and spont)
    n_per_step (int, list, or None): number of spike trains to generate at once
    p_dtype (str): datatype for spike probabilities (less than float32 will cause underflow
        due to low samplewise spike probabilities); does not determine output dtype
    p_noise_stddev (float, list, or None): standard deviation of Gaussian noise applied to
        spike probabilities on a per-sample basis (None or 0 specifies no noise)
    
    Returns
    -------
    tensor_spike_counts (tensor): same shape and dtype as tensor_rates
    """
    tensor_spike_probs = tf.multiply(tf.cast(tensor_rates, dtype=p_dtype), 1 / sr)
    if n_per_step is None:
        n_per_step = n_per_channel
    if p_noise_stddev is None:
        p_noise_stddev = 0
    if isinstance(n_per_channel, list):
        if len(tensor_spike_probs.shape) == 3:
            tensor_spike_probs = tf.expand_dims(tensor_spike_probs, -1)
        # If `n_per_channel` is a list, parallelize binomial sampling across
        # channels such that each channel can use a different `n`
        msg = "n_per_step={} not compatible with n_per_channel={}".format(n_per_step, n_per_channel)
        assert isinstance(n_per_step, list), msg
        assert len(n_per_step) == len(n_per_channel), msg
        assert all([(npc == 0) or (npc % nps == 0) for npc, nps in zip(n_per_channel, n_per_step)]), msg
        if not isinstance(p_noise_stddev, list):
            p_noise_stddev = [p_noise_stddev] * len(n_per_channel)
        assert len(p_noise_stddev) == len(n_per_channel), msg.replace('n_per_step', 'p_noise_stddev')
        perm = list(range(len(tensor_spike_probs.shape)))
        perm[0] = perm[-1]
        perm[-1] = 0
        tensor_spike_probs = tf.transpose(tensor_spike_probs, perm=perm)
        fn_output_signature = tf.TensorSpec(tensor_spike_probs[0].shape, tensor_spike_probs[0].dtype)
        tensor_spike_counts = tf.map_fn(
            map_fn_binomial,
            (tf.constant(n_per_channel), tf.constant(n_per_step), tensor_spike_probs, tf.constant(p_noise_stddev)),
            parallel_iterations=len(n_per_channel),
            swap_memory=False,
            infer_shape=False,
            fn_output_signature=fn_output_signature)
        tensor_spike_counts = tf.transpose(tensor_spike_counts, perm=perm)
    else:
        # Otherwise, call `map_fn_binomial` once with same `n` for all channels
        tensor_spike_counts = map_fn_binomial((n_per_channel, n_per_step, tensor_spike_probs, p_noise_stddev))
    return tf.cast(tensor_spike_counts, dtype=tensor_rates.dtype)
