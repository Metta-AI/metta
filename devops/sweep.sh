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
mkdir -p "${DATA_DIR}/sweep/$sweep"

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0

while true; do
  echo "[SWEEP:$sweep] Attempting rollout (consecutive failures: $consecutive_failures/$MAX_CONSECUTIVE_FAILURES)"

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
