#/bin/bash -e

args="${@:1}"

source ./devops/env.sh
uv run --active --directory mettagrid python setup.py build_ext --inplace

if [ -z "$NUM_CPUS" ]; then
    NUM_CPUS=$(lscpu | grep "CPU(s)" | awk '{print $NF}' | head -n1)
    NUM_CPUS=$((NUM_CPUS / 2))
fi
echo "NUM_CPUS: $NUM_CPUS"
NUM_GPUS=${NUM_GPUS:-1}
echo "NUM_GPUS: $NUM_GPUS"
NUM_NODES=${NUM_NODES:-1}
echo "NUM_NODES: $NUM_NODES"
MASTER_ADDR=${MASTER_ADDR:-localhost}
echo "MASTER_ADDR: $MASTER_ADDR"
MASTER_PORT=${MASTER_PORT:-12345}
echo "MASTER_PORT: $MASTER_PORT"
NODE_INDEX=${NODE_INDEX:-0}
echo "NODE_INDEX: $NODE_INDEX"

echo "Running train with args: $args"
PYTHONPATH=$PYTHONPATH:.

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --node-rank=$NODE_INDEX \
    tools/train.py \
    trainer.num_workers=$NUM_CPUS \
    wandb.enabled=true \
    wandb.track=true \
    $args
