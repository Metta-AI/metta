#/bin/bash -e

pkill -9 -f wandb
pkill -9 -f python
git pull
cd deps/puffergrid && git pull && cd ../..
cd deps/pufferlib && git pull && cd ../..
cd deps/fast_gae && git pull && cd ../..
cd deps/mettagrid && git pull && cd ../..
python -m tools.run \
    framework=pufferlib \
    cmd=train \
    hardware=pufferbox \
    wandb.track=true \
    "$@"
