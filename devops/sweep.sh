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
NUM_GPUS=${NUM_GPUS:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}
NODE_INDEX=${NODE_INDEX:-0}

echo "[INFO] Setting up sweep: $sweep_name"
mkdir -p "${DATA_DIR}/sweep/$sweep_name"

echo "[INFO] Starting sweep: $sweep_name..."

# Check if we should use distributed launch
if [ "$NUM_GPUS" -gt 1 ] || [ "$NUM_NODES" -gt 1 ]; then
  echo "[INFO] Launching distributed sweep with NUM_GPUS=$NUM_GPUS, NUM_NODES=$NUM_NODES"

  # Use torchrun to launch the sweep in distributed mode
  cmd="torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=1 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --node-rank=$NODE_INDEX \
    tools/sweep_rollout.py $args_for_rollout sweep_dir=$DATA_DIR/sweep/$sweep_name"
else
  # Single GPU/node - run directly
  cmd="tools/sweep_rollout.py $args_for_rollout sweep_dir=$DATA_DIR/sweep/$sweep_name"
fi

echo "[INFO] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep rollout failed: $sweep_name"
  exit 1
fi


