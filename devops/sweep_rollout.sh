#!/bin/bash
sweep="$1"
args="${@:2}"

DIST_ID=${DIST_ID:-localhost}
DIST_CFG_PATH=./train_dir/sweep/$sweep/dist_$DIST_ID.yaml

source ./devops/env.sh

mkdir -p ./train_dir/sweep/$sweep

echo "Sweep: $sweep == creating run..."
cmd="python -m tools.sweep_init sweep_name=$sweep dist_cfg_path=$DIST_CFG_PATH $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd

echo "Sweep: $sweep == training..."
cmd="./devops/train.sh dist_cfg_path=$DIST_CFG_PATH data_dir=./train_dir/sweep/$sweep/runs $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd

echo "Sweep: $sweep == evaluating..."
cmd="python -m tools.sweep_eval sweep_name=$sweep dist_cfg_path=$DIST_CFG_PATH data_dir=./train_dir/sweep/$sweep/runs $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd
exit $?
