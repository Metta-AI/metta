#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(command -v nvidia-smi > /dev/null && nvidia-smi --list-gpus | wc -l || echo 1)}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}
NODE_INDEX=${NODE_INDEX:-0}
TORCH_CRASH_TEST=${TORCH_CRASH_TEST:-false}

# Display configuration
echo "[CONFIG] Training configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Nodes: $NUM_NODES"
echo "  - Master address: $MASTER_ADDR"
echo "  - Master port: $MASTER_PORT"
echo "  - Node index: $NODE_INDEX"
echo "  - Crash test mode: $TORCH_CRASH_TEST"
if [[ "$TORCH_CRASH_TEST" == "true" ]]; then
  echo "  - Crash test timeout: 60s"
fi
echo "  - Arguments: $*"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export PYTHONOPTIMIZE=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting training..."

# run torchrun; preserve exit code and print a friendly line
set +e

if [[ "$TORCH_CRASH_TEST" == "true" ]]; then
  echo "[CRASH TEST] Running with timeout of 60 seconds..."
  timeout --signal=TERM --kill-after=10 60 uv run torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --node-rank=$NODE_INDEX \
    tools/run.py \
    "$@"
  EXIT_CODE=$?

  # If timeout killed the process, return 1 instead of 124
  if [[ $EXIT_CODE -eq 124 ]]; then
    echo "[CRASH TEST] Process killed after timeout"
    EXIT_CODE=1
  fi
else
  uv run torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --node-rank=$NODE_INDEX \
    tools/run.py \
    "$@"
  EXIT_CODE=$?
fi

set -e

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[SUCCESS] Training completed successfully"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
exit "$EXIT_CODE"
