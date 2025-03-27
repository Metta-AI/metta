#/bin/bash -e

args="${@:1}"

source ./devops/env.sh
./devops/checkout_and_build.sh

NUM_GPUS=${NUM_GPUS:-1}
echo "NUM_GPUS: $NUM_GPUS"
NUM_NODES=${NUM_NODES:-1}
echo "NUM_NODES: $NUM_NODES"
MASTER_ADDR=${MASTER_ADDR:-localhost}
echo "MASTER_ADDR: $MASTER_ADDR"
NODE_INDEX=${NODE_INDEX:-0}
echo "NODE_INDEX: $NODE_INDEX"

echo "Running train with args: $args"
PYTHONPATH=$PYTHONPATH:.

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --node-rank=$NODE_INDEX \
    tools/train.py \
    wandb.enabled=true \
    wandb.track=true \
    $args
