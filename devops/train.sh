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

git pull
# cd deps/pufferlib && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/fast_gae && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/mettagrid && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/wandb_carbs && git pull && cd ../..

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    tools/train.py \
    hardware=pufferbox \
    wandb.enabled=true \
    wandb.track=true \
    $args
