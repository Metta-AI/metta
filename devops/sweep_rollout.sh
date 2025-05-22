#!/bin/bash
# sweep_rollout.sh - Execute a single sweep rollout
set -e

# Parse arguments
sweep="$1"
args="${@:2}"

# Basic validation
if [ -z "$sweep" ]; then
  echo "[ERROR] Sweep name is required"
  exit 1
fi

source ./devops/setup.env

DIST_ID=${DIST_ID:-localhost}
DIST_CFG_PATH=./train_dir/sweep/$sweep/dist_$DIST_ID.yaml

echo "[INFO] Starting sweep rollout: $sweep"
mkdir -p ./train_dir/sweep/$sweep

# Initialize sweep
echo "[SWEEP:$sweep] Initializing sweep configuration..."
cmd="python -m tools.sweep_init sweep_name=$sweep dist_cfg_path=$DIST_CFG_PATH $args"
echo "[SWEEP:$sweep] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep initialization failed: $sweep"
  exit 1
fi

# Training phase
echo "[SWEEP:$sweep] Starting training phase..."
cmd="./devops/train.sh dist_cfg_path=$DIST_CFG_PATH data_dir=./train_dir/sweep/$sweep/runs $args"
echo "[SWEEP:$sweep] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Training failed for sweep: $sweep"
  exit 1
fi

# Evaluation phase
echo "[SWEEP:$sweep] Starting evaluation phase..."
cmd="python -m tools.sweep_eval sweep_name=$sweep dist_cfg_path=$DIST_CFG_PATH data_dir=./train_dir/sweep/$sweep/runs $args"
echo "[SWEEP:$sweep] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Evaluation failed for sweep: $sweep"
  exit 1
fi

echo "[SUCCESS] Sweep rollout completed: $sweep"
