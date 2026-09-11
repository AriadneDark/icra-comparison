#!/bin/bash

#############################
# Cluster specific settings #
#############################

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/captioning/job_%A_%a.out
#SBATCH -e logs/slurm/captioning/job_%A_%a.err

module load profile/deeplrn
module load python
source .venv/bin/activate

########################
# SGLang configuration #
########################

BASE_PORT=57000
PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID))
INPUT_FILE=$(printf "inputs/videos_%03d.txt" $SLURM_ARRAY_TASK_ID)
MODEL_NAME="Qwen/Qwen3.5-9B"

echo "Starting job $SLURM_ARRAY_TASK_ID on port $PORT"


# ---- Start SGLang server in background ----
singularity exec --nv --bind \
    .hf-cache:/hf_cache \
    --env HF_HOME=/hf_cache \
    --env HF_HUB_OFFLINE=1 sglang.sif \
    python -m sglang.launch_server --model-path $MODEL_NAME --host 127.0.0.1 --port $PORT > logs/server_${SLURM_ARRAY_TASK_ID}.log 2>&1 &

SERVER_PID=$!


# ---- Wait for server to be ready ----
echo "Waiting for server to be ready..."

for i in {1..60}; do
    if curl -s http://localhost:$PORT/v1/health > /dev/null; then
        echo "Server is up!"
        break
    fi
    sleep 5
done

# Fail if server never started
if ! curl -s http://localhost:$PORT/v1/models > /dev/null; then
    echo "Server failed to start"
    kill $SERVER_PID
    exit 1
fi


# ---- Processing ----
echo "Processing inputs..."

while IFS= read -r line || [ -n "$line" ]; do
    echo "Processing: $line"
    python main_sglang.py --video-path $line --output-path captions/qwen3.5_8b --port $PORT --model-name $MODEL_NAME
done < "$INPUT_FILE"


# ---- Cleanup ----
echo "Shutting down server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "Done."