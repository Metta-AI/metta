#/bin/bash -e

./devops/run.sh \
    cmd=train \
    wandb.track=true \
    "$@"
