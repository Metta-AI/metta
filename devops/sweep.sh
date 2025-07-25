#!/bin/bash
# sweep.sh - Launch sweep
set -e

# Parse arguments
args="${@:1}"

# Check for both run=...
has_run=$(echo "$args" | grep -E -c '(^|[[:space:]])run=' || true)

# Validate that exactly one is present
if [ "$has_run" -eq 0 ]; then
  echo "[ERROR] 'run' argument is required (e.g., run=my_sweep_name)"
  exit 1
fi

# Extract sweep name from run
sweep_name=$(echo "$args" | grep -E -o '(^|[[:space:]])run=[^ ]*' | sed 's/.*run=//')

# Replace run=<name> with sweep_name=<name> - handle both start of string and after space
args_for_rollout=$(echo "$args" | sed 's/^run=/sweep_name=/' | sed 's/ run=/ sweep_name=/g')
source ./devops/setup.env # TODO: Make sure that this is the right source-ing.

echo "[INFO] Setting up sweep: $sweep_name"
mkdir -p "${DATA_DIR}/sweep/$sweep_name"

echo "[INFO] Starting sweep: $sweep_name..."
cmd="tools/sweep_rollout.py $args_for_rollout"
echo "[INFO] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep rollout failed: $sweep_name"
  exit 1
fi


