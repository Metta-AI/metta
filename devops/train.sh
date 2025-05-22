
#!/bin/bash
# train.sh - Distributed training script
set -e

# Parse arguments
args="${@:1}"

source ./devops/setup.env

# System configuration
if [ -z "$NUM_CPUS" ]; then
  NUM_CPUS=$(lscpu | grep "CPU(s)" | awk '{print $NF}' | head -n1)
  NUM_CPUS=$((NUM_CPUS / 2))
fi

NUM_GPUS=${NUM_GPUS:-1}
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

PYTHONPATH=$PYTHONPATH:. torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/train.py \
  trainer.num_workers=$NUM_CPUS \
  wandb.enabled=true \
  $args

echo "[SUCCESS] Training completed successfully"
