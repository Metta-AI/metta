#/bin/bash -e

args="${@:1}"

export PYTHONUNBUFFERED=1
export WANDB_CONSOLE=off
export PYTHONPATH=$PYTHONPATH:.
export HYDRA_FULL_ERROR=1
echo "Running command: $cmd with args: $args"

#if NUM_GPUS is not set, set it to 1
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=1
fi

# same for NUM_NODES
if [ -z "$NUM_NODES" ]; then
    NUM_NODES=1
fi

# same for MASTER_ADDR
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR="localhost"
fi

# same for NODE_RANK
if [ -z "$NODE_RANK" ]; then
    NODE_RANK=0
fi

git pull
# cd deps/pufferlib && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/fast_gae && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/mettagrid && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/wandb_carbs && git pull && cd ../..

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPUS \
    --master-addr=$MASTER_ADDR \
    --node-rank=$NODE_RANK \
    tools/train.py \
    hardware=pufferbox \
    wandb.enabled=true \
    wandb.track=true \
    $args
