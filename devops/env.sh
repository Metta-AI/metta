#!/bin/bash
# env.sh - Environment configuration
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONOPTIMIZE=0
export HYDRA_FULL_ERROR=1
export WANDB_CONSOLE=off
export WANDB_DIR="./wandb"
