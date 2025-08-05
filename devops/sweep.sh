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

# Set up heartbeat monitoring (similar to train.sh)
WANDB_DIR=${DATA_DIR}/sweep/$sweep_name
HEARTBEAT_FILE=${HEARTBEAT_FILE:-$WANDB_DIR/heartbeat.txt}
HEARTBEAT_TIMEOUT=${HEARTBEAT_TIMEOUT:-600}
mkdir -p "$(dirname "$HEARTBEAT_FILE")"

# Start heartbeat monitor in background
echo "[INFO] Starting heartbeat monitor with timeout ${HEARTBEAT_TIMEOUT}s for file $HEARTBEAT_FILE"
python -m metta.common.util.heartbeat monitor "$HEARTBEAT_FILE" --pid $$ --timeout "$HEARTBEAT_TIMEOUT" &
HEARTBEAT_PID=$!

# Export for child processes
export HEARTBEAT_FILE

# Ensure monitor is killed on exit
trap "kill $HEARTBEAT_PID 2>/dev/null || true" EXIT

echo "[INFO] Starting sweep: $sweep_name..."
cmd="tools/sweep_rollout.py $args_for_rollout sweep_dir=$DATA_DIR/sweep/$sweep_name"
echo "[INFO] Running: $cmd"
if ! $cmd; then
  echo "[ERROR] Sweep rollout failed: $sweep_name"
  exit 1
fi


