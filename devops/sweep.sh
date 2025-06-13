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

# Extract rollout_count from arguments (optional)
rollout_count=$(echo "$args" | grep -o '\-\-rollout-count=[^ ]*' | sed 's/--rollout-count=//')
if [ -z "$rollout_count" ]; then
  rollout_count=999999  # Default to very high number (effectively infinite)
fi

# Validate rollout_count is a number
if ! [[ "$rollout_count" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] rollout_count must be a positive integer"
  exit 1
fi

# Remove --rollout-count from args before passing to sweep_rollout.sh
args=$(echo "$args" | sed 's/--rollout-count=[^ ]*//')

source ./devops/setup.env

echo "[INFO] Starting continuous sweep execution: $sweep"
echo "[INFO] Rollout limit: $rollout_count"
mkdir -p "${DATA_DIR}/sweep/$sweep"

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
