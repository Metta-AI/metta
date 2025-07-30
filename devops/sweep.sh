#!/bin/bash
# sweep.sh - Continuous sweep execution with retry logic
set -e

# Parse arguments
args="${@:1}"

# Check for both run=...
has_run=$(echo "$args" | grep -E -c '(^|[[:space:]])run=' || true)

# Validate that exactly one is present
if [ "$has_run" -eq 0 ]; then
  echo "[ERROR] Either 'run' argument is required (e.g., run=my_sweep_name)"
  exit 1
fi

# Extract sweep name from run
sweep_name=$(echo "$args" | grep -E -o '(^|[[:space:]])run=[^ ]*' | sed 's/.*run=//')


# Replace run=<name> with sweep_name=<name> - handle both start of string and after space
args_for_rollout=$(echo "$args" | sed 's/^run=/sweep_name=/' | sed 's/ run=/ sweep_name=/g')

source ./devops/setup.env # TODO: Make sure that this is the right source-ing.

echo "[INFO] Setting up sweep: $sweep_name"
mkdir -p "${DATA_DIR}/sweep/$sweep_name"

# Initialize the sweep
# This script either creates or fetches sweep details
# and ensures the data is written to a local $SWEEP_DIR/metadata.yaml
echo "[SWEEP:$sweep_name] Initializing sweep configuration..."
cmd="tools/sweep_setup.py sweep_name=$sweep_name"
echo "[SWEEP:$sweep_name] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep initialization failed: $sweep_name"
  exit 1
fi

# Retry configuration
MAX_CONSECUTIVE_FAILURES=3
consecutive_failures=0

while true; do
  if ./devops/sweep_rollout.sh $args_for_rollout; then
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))

    if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
      echo "[ERROR] Maximum consecutive failures reached ($MAX_CONSECUTIVE_FAILURES), terminating sweep: $sweep_name"
      exit 1
    fi

    sleep 5
  fi
done
