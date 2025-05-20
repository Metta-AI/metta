#!/bin/bash
args="${@:1}"

sweep=$(echo "$args" | grep -o 'run=[^ ]*' | sed 's/run=//')
if [ -z "$sweep" ]; then
  echo "Error: run arg is required"
  exit 1
fi

source ./devops/env.sh
<<<<<<< HEAD
./devops/build_mettagrid.sh
=======
uv run --active --directory mettagrid python setup.py build_ext --inplace
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87

mkdir -p ./train_dir/sweep/$sweep
# keep running sweep until it fails N consecutive times
N=3
consecutive_failures=0
while true; do
  echo "Running sweep_rollout for $sweep with $consecutive_failures consecutive failures"
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
