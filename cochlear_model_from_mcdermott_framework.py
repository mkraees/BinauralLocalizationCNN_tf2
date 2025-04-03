import os
import numpy as np
import json
import copy
import time
import util_cochlea
from joblib import Parallel, delayed


def cochlea_processing(audio_path):
    dur = 2  # 2 seconds
    sr = 48000

    audio_data = np.load(audio_path, allow_pickle=True)

    ## TODO: check why certain files are not 96000
    if audio_data.shape[0] == 96000:

        y = audio_data[:sr].reshape(1, sr, dur)

        kwargs_cochlea = {
            'config_filterbank': {'max_hi': 20000.0, 'min_lo': 30.0, 'mode': 'half_cosine_filterbank', 'num_cf': 39},
            'config_subband_processing': {'power_compression': 0.3, 'rectify': True},
            'kwargs_fir_lowpass_filter_input': {},
            'kwargs_fir_lowpass_filter_output': {'cutoff': 4000, 'numtaps': 4097, 'window': ['kaiser', 5.0]},
            'sr_cochlea': 48000,
            'sr_input': 48000,
            'sr_output': 8000
        }

        # Cochlear model for ear index 0
        y0, _ = util_cochlea.cochlea(y[..., 0], **copy.deepcopy(kwargs_cochlea))
        # Cochlear model for ear index 1
        y1, _ = util_cochlea.cochlea(y[..., 1], **copy.deepcopy(kwargs_cochlea))

        y_out = np.concatenate([y0.numpy(), y1.numpy()], axis=0)
        return y_out
    else:
        print(f"File: {audio_path}, shape: {audio_data.shape} \n\n")


audio_path = '/mnt/lustre/work/macke/mwe234/datasets/simulated/500_spatial_360_7'
output_dir = '/mnt/lustre/work/macke/mwe234/datasets/simulated/cochlea_500_spatial_360_7'
os.makedirs(output_dir, exist_ok=True)

audio_files = [f for f in os.listdir(audio_path) if f.endswith('.npy')]

def process_audio(file):
    input_file = os.path.join(audio_path, file)
    output_file = os.path.join(output_dir, file)
    
    # print(f"Processing {file} \n\n")
    y_out = cochlea_processing(input_file)
    np.save(output_file, y_out)

# Measure time
start_time = time.time()

# Parallel processing using joblib
num_cores = os.cpu_count()  # Get number of available cores
Parallel(n_jobs=num_cores)(delayed(process_audio)(file) for file in audio_files)

end_time = time.time()
total_time = (end_time - start_time) / 60
print(f"Total processing time: {total_time:.2f} minutes")