#!/bin/bash
# sweep.sh - Launch sweep
set -e

# Parse arguments
args="${@:1}"

# Check for both run=...
has_run=$(echo "$args" | grep -E -c '(^|[[:space:]])run=' || true)

# Validate that exactly one is present
if [ "$has_run" -eq 0 ]; then
  echo "[ERROR] 'run' argument is required (e.g., run=my_sweep_name)"
  exit 1
fi

# Extract sweep name from run
sweep_name=$(echo "$args" | grep -E -o '(^|[[:space:]])run=[^ ]*' | sed 's/.*run=//')

# Replace run=<name> with sweep_name=<name> - handle both start of string and after space
args_for_rollout=$(echo "$args" | sed 's/^run=/sweep_name=/' | sed 's/ run=/ sweep_name=/g')
source ./devops/setup.env # TODO: Make sure that this is the right source-ing.

# Set default values for distributed training if not set
# Auto-detect number of GPUs if not explicitly set
if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "[INFO] Auto-detected $NUM_GPUS GPUs"
    else
        NUM_GPUS=1
    fi
fi

NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}
NODE_INDEX=${NODE_INDEX:-0}

# Export for child processes
export NUM_GPUS NUM_NODES MASTER_ADDR MASTER_PORT NODE_INDEX

echo "[INFO] Setting up sweep: $sweep_name"
mkdir -p "${DATA_DIR}/sweep/$sweep_name"

echo "[INFO] Starting sweep: $sweep_name..."

set +e
PYTHONPATH=$PYTHONPATH:. uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/sweep_rollout.py \
  $args_for_rollout \
  sweep_dir=$DATA_DIR/sweep/$sweep_name
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[SUCCESS] Sweep completed successfully"
else
  echo "[ERROR] Sweep failed with exit code $EXIT_CODE"
fi
exit "$EXIT_CODE"


