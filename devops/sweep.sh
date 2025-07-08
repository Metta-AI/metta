#!/bin/bash
# sweep.sh - Continuous sweep execution with retry logic
set -e

# Parse arguments
args="${@:1}"

# Check for both run= and sweep_run= parameters
# Use precise patterns to avoid matching sweep_run= when looking for run=
has_run=$(echo "$args" | grep -E -c '(^|[[:space:]])run=' || true)
has_sweep_run=$(echo "$args" | grep -c 'sweep_run=' || true)

# Validate that exactly one is present
if [ "$has_run" -eq 0 ] && [ "$has_sweep_run" -eq 0 ]; then
  echo "[ERROR] Either 'run' or 'sweep_run' argument is required (e.g., run=my_sweep_name or sweep_run=my_sweep_name)"
  exit 1
elif [ "$has_run" -gt 0 ] && [ "$has_sweep_run" -gt 0 ]; then
  echo "[ERROR] Cannot specify both 'run' and 'sweep_run' arguments. Please use only one."
  exit 1
fi

# Extract sweep name from whichever parameter is present
if [ "$has_run" -gt 0 ]; then
  sweep_run=$(echo "$args" | grep -E -o '(^|[[:space:]])run=[^ ]*' | sed 's/.*run=//')
else
  sweep_run=$(echo "$args" | grep -o 'sweep_run=[^ ]*' | sed 's/sweep_run=//')
fi

# Convert run= to sweep_run= for downstream scripts, ensuring only sweep_run= is passed
if [ "$has_run" -gt 0 ]; then
  # Replace run= with sweep_run= - handle both start of string and after space
  args_for_rollout=$(echo "$args" | sed 's/^run=/sweep_run=/' | sed 's/ run=/ sweep_run=/g')
else
  # If already using sweep_run=, pass through as-is
  args_for_rollout="$args"
fi

source ./devops/setup.env

echo "[INFO] Starting sweep: $sweep_run"
mkdir -p "${DATA_DIR}/sweep/$sweep_run"

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0

while true; do
  if ./devops/sweep_rollout.sh $args_for_rollout; then
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
