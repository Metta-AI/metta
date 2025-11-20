#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(command -v nvidia-smi > /dev/null && nvidia-smi --list-gpus | wc -l || echo 1)}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}
NODE_INDEX=${NODE_INDEX:-0}

# Display configuration
echo "[CONFIG] Training configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Nodes: $NUM_NODES"
echo "  - Master address: $MASTER_ADDR"
echo "  - Master port: $MASTER_PORT"
echo "  - Node index: $NODE_INDEX"
echo "  - Arguments: $*"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export PYTHONOPTIMIZE=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting training..."

# Set up a single log file for Datadog and debugging
TRAINING_LOG_DIR="/tmp/training_logs"
mkdir -p "$TRAINING_LOG_DIR"
chmod 777 "$TRAINING_LOG_DIR"

TRAINING_COMBINED_LOG="$TRAINING_LOG_DIR/training_combined.log"
touch "$TRAINING_COMBINED_LOG"
# Ensure file is readable by dd-agent user (Datadog agent runs as dd-agent)
chmod 666 "$TRAINING_COMBINED_LOG"
# Also ensure directory is executable by all (needed for dd-agent to access the file)
chmod o+x "$TRAINING_LOG_DIR" 2>/dev/null || true
# If running as root, try to set ownership to dd-agent if it exists
if [ "$(id -u)" = "0" ] && id dd-agent >/dev/null 2>&1; then
  chown dd-agent:dd-agent "$TRAINING_COMBINED_LOG" 2>/dev/null || true
  chown dd-agent:dd-agent "$TRAINING_LOG_DIR" 2>/dev/null || true
fi

echo "[INFO] Logging training output to: $TRAINING_COMBINED_LOG"

# run torchrun; preserve exit code and print a friendly line
set +e
uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/run.py \
  "$@" 2>&1 | tee -a "$TRAINING_COMBINED_LOG"
EXIT_CODE=${PIPESTATUS[0]}   # real torchrun exit code
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[SUCCESS] Training completed successfully"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
exit "$EXIT_CODE"
