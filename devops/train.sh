#!/bin/bash
# train.sh - Distributed training script
set -e

# Parse arguments
args="${@:1}"

source ./devops/setup.env

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

# System configuration
if [ -z "$NUM_CPUS" ]; then
  if command -v lscpu &> /dev/null; then
    # Linux
    NUM_CPUS=$(lscpu | grep "CPU(s)" | awk '{print $NF}' | head -n1)
    NUM_CPUS=$((NUM_CPUS / 2))
  elif command -v sysctl &> /dev/null; then
    # macOS
    NUM_CPUS=$(sysctl -n hw.ncpu)
    NUM_CPUS=$((NUM_CPUS / 2))
  else
    NUM_CPUS=1  # fallback
  fi
fi

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
echo "  - CPUs: $NUM_CPUS"
echo "  - GPUs: $NUM_GPUS"
echo "  - Nodes: $NUM_NODES"
echo "  - Master address: $MASTER_ADDR"
echo "  - Master port: $MASTER_PORT"
echo "  - Node index: $NODE_INDEX"
echo "  - Arguments: $args"

echo "[INFO] Starting distributed training..."

PYTHONPATH=$PYTHONPATH:. uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/train.py \
  trainer.num_workers=$((NUM_CPUS / NUM_GPUS)) \
  wandb.enabled=true \
  $args

echo "[SUCCESS] Training completed successfully"
