#!/bin/bash
sweep="$1"
args="${@:2}"

source ./devops/env.sh

mkdir -p ./train_dir/sweep/$sweep

echo "Sweep: $sweep == creating run..."
cmd="python -m tools.sweep_init sweep_name=$sweep dist_cfg_path=$dist_cfg_path $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd

echo "Sweep: $sweep == training..."
cmd="./devops/train.sh dist_cfg_path=$dist_cfg_path data_dir=./train_dir/sweep/$sweep/runs $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd

echo "Sweep: $sweep == evaluating..."
cmd="python -m tools.sweep_eval dist_cfg_path=$dist_cfg_path data_dir=./train_dir/sweep/$sweep/runs $args"
echo "Sweep: $sweep == running command: $cmd"
$cmd
