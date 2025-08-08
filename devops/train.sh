#!/bin/bash
# train.sh - Distributed training script
set -e

# Parse arguments
args="${@:1}"

# Start heartbeat monitor if available
HEARTBEAT_FILE=${HEARTBEAT_FILE:-$WANDB_DIR/heartbeat.txt}
HEARTBEAT_TIMEOUT=${HEARTBEAT_TIMEOUT:-600} # Read from env or default to 600

if [ "$HEARTBEAT_TIMEOUT" -ne 0 ]; then
  echo "[INFO] Starting heartbeat monitor with timeout ${HEARTBEAT_TIMEOUT}s for file $HEARTBEAT_FILE"
  python -m metta.common.util.heartbeat monitor "$HEARTBEAT_FILE" --pid $$ --timeout "$HEARTBEAT_TIMEOUT" &
  HEARTBEAT_PID=$!
  trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT
else
  echo "[INFO] Heartbeat monitor deactivated (timeout is 0)."
fi
export HEARTBEAT_FILE


# Auto-detect GPUs if not set
if [ -z "$NUM_GPUS" ]; then
  if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
  else
    NUM_GPUS=1
  fi
fi

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
echo "  - Arguments: $args"

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting training..."

set +e
PYTHONPATH=$PYTHONPATH:. uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/train.py \
  trainer.num_workers=null \
  $args
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[SUCCESS] Training completed successfully"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE"
fi
exit "$EXIT_CODE"
