#/bin/bash -e

args="${@:1}"

echo "Running command: $cmd with args: $args"

export PYTHONUNBUFFERED=1
export WANDB_CONSOLE=off
export PYTHONPATH=$PYTHONPATH:.
export HYDRA_FULL_ERROR=1
export NUM_GPUS=${NUM_GPUS:-1}
export NUM_NODES=${NUM_NODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export NODE_RANK=${NODE_RANK:-0}

if [ -z "$SKIP_BUILD" ] || [ "$SKIP_BUILD" = "0" ]; then
    git pull
    # cd deps/pufferlib && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/fast_gae && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/mettagrid && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/wandb_carbs && git pull && cd ../..
fi

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
