#!/bin/bash
args="${@:1}"

sweep=$(echo "$args" | grep -o 'run=[^ ]*' | sed 's/run=//')
if [ -z "$sweep" ]; then
    echo "Error: run arg is required"
    exit 1
fi

source ./devops/env.sh
./devops/checkout_and_build.sh

mkdir -p ./train_dir/sweep/$sweep
# keep running sweep until it fails N consecutive times
N=10
consecutive_failures=0
while true; do
    ./devops/sweep_rollout.sh $sweep $args
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
