#!/usr/bin/env bash

set -euo pipefail

cd /workspace/metta

# Ensure we are using the project virtualenv
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2> /dev/null || true
fi
. .venv/bin/activate

export WRAPPER_PID=$BASHPID

echo "[SWEEP] Configuring environment..."
bash ./devops/skypilot/config/lifecycle/configure_environment.sh

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
source "$METTA_ENV_FILE"

echo "[SWEEP] Running CUDA diagnostics..."
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "[SWEEP] nvidia-smi command failed (above)."
  fi
else
  echo "[SWEEP] nvidia-smi not found on PATH."
fi

export RANK=${SKYPILOT_NODE_RANK:-0}
export IS_HEAD=$([[ "$RANK" == "0" ]] && echo "true" || echo "false")
export IS_MASTER="$IS_HEAD"  # For compatibility with monitor_utils.sh
HEAD_IP=$(echo "${SKYPILOT_NODE_IPS:-127.0.0.1}" | head -n1)
RAY_PORT=${RAY_HEAD_PORT:-6379}
RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-10001}

# Set up per-node heartbeat monitoring
export START_TIME=$(date +%s)
# Create per-node heartbeat file using node rank to ensure isolation
export HEARTBEAT_FILE="${JOB_METADATA_DIR}/heartbeat_node_${RANK}"
echo "[SWEEP] Per-node heartbeat file: $HEARTBEAT_FILE"

# Detect GPUs and CPUs on this node (not provided by SkyPilot)
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
else
  GPU_COUNT=0
fi

if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "[SWEEP] Warning: No GPUs detected, defaulting Ray to CPU-only mode."
fi

echo "[SWEEP] Node rank: $RANK (head: $IS_HEAD)"
echo "[SWEEP] Head IP: $HEAD_IP"
echo "[SWEEP] GPUs per node: $GPU_COUNT"
echo "[SWEEP] Ray ports: tcp=$RAY_PORT client=$RAY_CLIENT_PORT"

# Export detected hardware counts for Python processes to use
export METTA_DETECTED_GPUS_PER_NODE="$GPU_COUNT"

start_ray_head() {
  echo "[SWEEP] Starting Ray head node..."
  ray start \
    --head \
    --port "$RAY_PORT" \
    --ray-client-server-port "$RAY_CLIENT_PORT" \
    --dashboard-host "0.0.0.0" \
    --disable-usage-stats \
    --num-gpus "$GPU_COUNT"
  sleep 5
}

start_ray_worker() {
  echo "[SWEEP] Starting Ray worker..."
  until ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats --num-gpus "$GPU_COUNT" ; do
    echo "[SWEEP] Worker waiting for head..."
    sleep 5
  done
  sleep 5
}

start_heartbeat_monitor() {
  if [[ -n "${HEARTBEAT_TIMEOUT:-}" ]]; then
    echo "[SWEEP] Starting per-node heartbeat monitor (timeout: ${HEARTBEAT_TIMEOUT}s)..."
    bash ./devops/skypilot/config/monitors/heartbeat_monitor.sh &
    HEARTBEAT_PID=$!
    echo "[SWEEP] Started heartbeat monitor with PID $HEARTBEAT_PID for node $RANK"
  else
    echo "[SWEEP] Heartbeat monitoring disabled (HEARTBEAT_TIMEOUT not set)"
  fi
}

if [[ "$IS_HEAD" == "true" ]]; then
  start_ray_head
  # Use local mode address instead of client mode for better GPU allocation
  # This format allows Ray to properly allocate GPUs to worker processes
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  echo "[SWEEP] Ray address:${RAY_ADDRESS}"
else
  start_ray_worker
fi

# Start per-node heartbeat monitor after Ray is up
start_heartbeat_monitor

# Set up cleanup handler
cleanup() {
  echo "[SWEEP] Cleaning up node $RANK..."
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill $HEARTBEAT_PID 2>/dev/null || true
  fi
  ray stop || true
  exit 0
}
trap cleanup EXIT INT TERM

MODULE_PATH=${SWEEP_MODULE_PATH:?missing SWEEP_MODULE_PATH}
MODULE_ARGS=${SWEEP_MODULE_ARGS:-}

if [[ "$IS_HEAD" == "true" ]]; then
  echo "[SWEEP] Launching tool: ${MODULE_PATH} ${MODULE_ARGS}"
  # Update heartbeat file periodically while running the sweep
  (
    while true; do
      echo "$(date +%s)" > "$HEARTBEAT_FILE"
      sleep 30
    done
  ) &
  HEARTBEAT_UPDATER_PID=$!

  uv run tools/run.py "${MODULE_PATH}" ${MODULE_ARGS}

  # Stop heartbeat updater when done
  kill $HEARTBEAT_UPDATER_PID 2>/dev/null || true
else
  echo "[SWEEP] Worker node entering idle loop with heartbeat updates."
  # Worker nodes also update their heartbeat files
  while true; do
    echo "$(date +%s)" > "$HEARTBEAT_FILE"
    sleep 30
  done
fi
