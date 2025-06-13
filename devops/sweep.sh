#!/bin/bash
# sweep.sh - Continuous sweep execution with retry logic
set -e

# Parse arguments
args="${@:1}"

# Extract and validate sweep name
sweep_run=$(echo "$args" | grep -o 'sweep_run=[^ ]*' | sed 's/sweep_run=//')
if [ -z "$sweep_run" ]; then
  echo "[ERROR] 'sweep_run' argument is required (e.g., sweep_run=my_sweep_name)"
  exit 1
fi

source ./devops/setup.env

echo "[INFO] Starting continuous sweep execution: $sweep_run"
mkdir -p "${DATA_DIR}/sweep/$sweep_run"

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0

while true; do
  echo "[SWEEP:$sweep_run] Attempting rollout (consecutive failures: $consecutive_failures/$MAX_CONSECUTIVE_FAILURES)"

  if ./devops/sweep_rollout.sh $args; then
    echo "[SUCCESS] Sweep rollout completed successfully: $sweep_run"
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))
    echo "[WARNING] Sweep rollout failed (failure $consecutive_failures/$MAX_CONSECUTIVE_FAILURES): $sweep_run"

    if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
      echo "[ERROR] Maximum consecutive failures reached ($MAX_CONSECUTIVE_FAILURES), terminating sweep: $sweep_run"
      exit 1
    fi

    echo "[INFO] Retrying sweep rollout in 5 seconds..."
    sleep 5
  fi
done

echo "[SUCCESS] Sweep completed!"
