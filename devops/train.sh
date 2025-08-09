#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(command -v nvidia-smi >/dev/null && nvidia-smi --list-gpus | wc -l || echo 1)}
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
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting training..."

# run torchrun; preserve exit code and print a friendly line
set +e
uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/train.py \
  trainer.num_workers=null \
  "$@"
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[SUCCESS] Training completed successfully"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
exit "$EXIT_CODE"
