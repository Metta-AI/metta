#!/bin/bash
# sweep.sh - Launch sweep with simplified pipeline
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

# TODO: review desired cmd ENV settings
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting sweep: $sweep_name..."

# Run the simplified sweep rollout
# No special directory setup needed - everything is passed as command-line args
cmd="tools/sweep_execute.py data_dir=$DATA_DIR $args_for_rollout"
echo "[INFO] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep rollout failed: $sweep_name"
  exit 1
fi
