#!/bin/bash -e
sweep="$1"
args="${@:2}"

source ./devops/env.sh
./devops/checkout_and_build.sh

mkdir -p ./train_dir/sweep/$sweep
# keep running sweep until it fails N consecutive times
N=10
consecutive_failures=0
while true; do

    echo "Sweep: $sweep == creating run..."
    tmp_run=$(mktemp -p ./train_dir/sweep/$sweep/)
    cmd="python -m tools.sweep_init sweep_name=$sweep +save_run=$tmp_run $args"
    echo "Sweep: $sweep == running command: $cmd"
    $cmd
    run_id=$(cat $tmp_run)

    echo "Sweep: $sweep == running run $run_id..."
    ./devops/train.sh data_dir=./train_dir/sweep/$sweep/runs run=$run_id $args

    cmd="python -m tools.sweep_eval sweep_name=$sweep run=$run_id $args"
    echo "Sweep: $sweep == running command: $cmd"
    $cmd

    if [ $? -ne 0 ]; then
        consecutive_failures=$((consecutive_failures + 1))
        if [ $consecutive_failures -ge $N ]; then
            echo "Sweep failed $N consecutive times, exiting"
            exit 1
        fi
    else
        consecutive_failures=0
    fi
done
