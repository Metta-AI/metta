#!/bin/bash
# sweep_rollout.sh - Execute a single sweep rollout
set -e

# Parse arguments
args="${@:1}"

# Extract and validate sweep run
sweep_run=$(echo "$args" | grep -o 'sweep_run=[^ ]*' | sed 's/sweep_run=//')
if [ -z "$sweep_run" ]; then
  echo "[ERROR] 'sweep_run' argument is required (e.g., sweep_run=my_sweep_name)"
  exit 1
fi

source ./devops/setup.env

DIST_ID=${DIST_ID:-localhost}
DIST_CFG_PATH="$DATA_DIR/sweep/$sweep_run/dist_$DIST_ID.yaml"

echo "[INFO] Starting sweep rollout: $sweep_run"
mkdir -p "$DATA_DIR/sweep/$sweep_run"

# Initialize sweep
echo "[SWEEP:$sweep_run] Initializing sweep configuration..."
cmd="./tools/sweep_init.py dist_cfg_path=$DIST_CFG_PATH $args"
echo "[SWEEP:$sweep_run] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep initialization failed: $sweep_run"
  exit 1
fi

# Training phase - use train_job config
echo "[SWEEP:$sweep_run] Starting training phase..."
cmd="./devops/train.sh dist_cfg_path=$DIST_CFG_PATH data_dir=$DATA_DIR/sweep/$sweep_run/runs"
echo "[SWEEP:$sweep_run] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Training failed for sweep: $sweep_run"
  exit 1
fi

# Evaluation phase
echo "[SWEEP:$sweep_run] Starting evaluation phase..."
cmd="./tools/sweep_eval.py dist_cfg_path=$DIST_CFG_PATH data_dir=$DATA_DIR/sweep/$sweep_run/runs $args"
echo "[SWEEP:$sweep_run] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Evaluation failed for sweep: $sweep_run"
  exit 1
fi

echo "[SUCCESS] Sweep rollout completed: $sweep_run"
