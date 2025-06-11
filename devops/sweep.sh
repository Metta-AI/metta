#!/bin/bash
# sweep.sh - Continuous sweep execution with retry logic
set -e

# Parse arguments
args="${@:1}"

# Extract and validate sweep name
sweep=$(echo "$args" | grep -o 'run=[^ ]*' | sed 's/run=//')
if [ -z "$sweep" ]; then
  echo "[ERROR] 'run' argument is required (e.g., run=my_sweep_name)"
  exit 1
fi

source ./devops/setup.env

echo "[INFO] Starting continuous sweep execution: $sweep"
mkdir -p ./train_dir/sweep/$sweep

# Extract sweep_params to check for rollout_count
sweep_params=$(echo "$args" | grep -o '++sweep_params=[^ ]*' | sed 's/++sweep_params=//')
rollout_count=""

if [ -n "$sweep_params" ]; then
  config_file="configs/${sweep_params}.yaml"
  if [ -f "$config_file" ]; then
    rollout_count=$(grep "^rollout_count:" "$config_file" | sed 's/rollout_count: *//' | sed 's/ *#.*//')
  fi
fi

# Set default rollout count if not specified
if [ -z "$rollout_count" ] || [ "$rollout_count" = "" ]; then
  rollout_count=999999  # Default to very high number (infinite)
fi

echo "[INFO] Rollout limit: $rollout_count"

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0
rollout_number=0

while [ $rollout_number -lt $rollout_count ]; do
  rollout_number=$((rollout_number + 1))
  echo "[SWEEP:$sweep] Attempting rollout $rollout_number/$rollout_count (consecutive failures: $consecutive_failures/$MAX_CONSECUTIVE_FAILURES)"

  if ./devops/sweep_rollout.sh $sweep $args; then
    echo "[SUCCESS] Sweep rollout completed successfully: $sweep"
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))
    echo "[WARNING] Sweep rollout failed (failure $consecutive_failures/$MAX_CONSECUTIVE_FAILURES): $sweep"

    if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
      echo "[ERROR] Maximum consecutive failures reached ($MAX_CONSECUTIVE_FAILURES), terminating sweep: $sweep"
      exit 1
    fi

    echo "[INFO] Retrying sweep rollout in 5 seconds..."
    sleep 5
  fi
done

echo "[SUCCESS] Sweep completed! Finished $rollout_count rollouts for: $sweep"
