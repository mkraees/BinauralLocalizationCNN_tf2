#!/bin/bash
#SBATCH --partition=cpu-galvani
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=16        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --mem=64G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:0
#SBATCH --output=./cochlear__%j.log   # Standard output to log file with job ID
#SBATCH --error=./cochlear__%j.err    # Error output to log file with job ID

# Run the Python script inside the conda environment
conda run -n BinauralLocalizationCNN_tf2_env python3 -u cochlear_model_from_mcdermott_framework.py