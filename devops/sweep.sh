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

echo "[INFO] Starting sweep: $sweep_run"
mkdir -p "${DATA_DIR}/sweep/$sweep_run"

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0

while true; do
  if ./devops/sweep_rollout.sh $args; then
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))

    if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
      echo "[ERROR] Maximum consecutive failures reached ($MAX_CONSECUTIVE_FAILURES), terminating sweep: $sweep_run"
      exit 1
    fi

    sleep 5
  fi
done
