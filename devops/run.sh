#/bin/bash -e

cmd="$1"
args="${@:2}"

export PYTHONUNBUFFERED=1

echo "Running command: $cmd with args: $args"

git pull
cd deps/puffergrid && git pull && python setup.py build_ext --inplace && cd ../..
# cd deps/pufferlib && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/fast_gae && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/mettagrid && git pull && python setup.py build_ext --inplace && cd ../..
cd deps/wandb_carbs && git pull && cd ../..

HYDRA_FULL_ERROR=1 python -m tools.$cmd \
    hardware=pufferbox \
    wandb.enabled=true \
    wandb.track=true $args
