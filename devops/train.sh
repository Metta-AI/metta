#/bin/bash -e

args="${@:1}"

./devops/env.sh
./devops/checkout_and_build.sh

NUM_GPUS=${NUM_GPUS:-1}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
NODE_RANK=${NODE_RANK:-0}


echo "Running train with args: $args"
PYTHONPATH=$PYTHONPATH:.

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --node-rank=$NODE_RANK \
    tools/train.py \
    wandb.enabled=true \
    wandb.track=true \
    $args
