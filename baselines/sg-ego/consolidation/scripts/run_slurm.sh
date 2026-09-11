#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/grounding/job_%A_%a.out
#SBATCH -e logs/slurm/grounding/job_%A_%a.err

# Cluster-specific commands
module load python
module load profile/deeplrn
# Activate the venv or conda environment
source .venv/bin/activate

# Optionally set the HF_HOME environment variable to a local cache directory to avoid downloading models multiple times
# export HF_HOME=.hf-cache
# export HF_HUB_OFFLINE=1

INPUT_FILE=data/example/input_files/windows.txt

# If you want to parallelize the job you split the input file into multiple files with:
#   split -n l/128 -d --additional-suffix=.txt data/example/input_files/windows.txt data/example/input_files/windows_
#   sbatch --array=0-127 consolidation/scripts/run_slurm.sh
#
# Then uncomment the following line
# INPUT_FILE=$(printf "data/example/input_files/windows_%03d.txt" $SLURM_ARRAY_TASK_ID)

python -m consolidation.main --root data/example/ --job-file $INPUT_FILE --captions-version qwen3.5_9b --frame-graphs-version v1