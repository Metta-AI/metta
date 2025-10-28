#!/usr/bin/env bash

set -euo pipefail

cd /workspace/metta

# Ensure we are using the project virtualenv
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2> /dev/null || true
fi
. .venv/bin/activate

echo "Setting up Python environment..."
uv sync

echo "[SIMPLE] Configuring environment..."
bash ./devops/skypilot/config/lifecycle/configure_environment.sh

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
source "$METTA_ENV_FILE"

export RANK=${SKYPILOT_NODE_RANK:-0}
export IS_HEAD=$([[ "$RANK" == "0" ]] && echo "true" || echo "false")
HEAD_IP=$(echo "${SKYPILOT_NODE_IPS:-127.0.0.1}" | head -n1)
RAY_PORT=${RAY_HEAD_PORT:-6379}
RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-10001}

echo "[SIMPLE] Node rank: $RANK (head: $IS_HEAD)"
echo "[SIMPLE] Head IP: $HEAD_IP"
echo "[SIMPLE] Ray ports: tcp=$RAY_PORT client=$RAY_CLIENT_PORT"

start_ray_head() {
  echo "[SIMPLE] Starting Ray head node..."
  ray start \
    --head \
    --port "$RAY_PORT" \
    --ray-client-server-port "$RAY_CLIENT_PORT" \
    --dashboard-host "0.0.0.0" \
    --disable-usage-stats
  sleep 5
}

start_ray_worker() {
  echo "[SIMPLE] Starting Ray worker..."
  until ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats; do
    echo "[SIMPLE] Worker waiting for head..."
    sleep 5
  done
  sleep 5
}

if [[ "$IS_HEAD" == "true" ]]; then
  start_ray_head
  export RAY_ADDRESS="ray://${HEAD_IP}:${RAY_CLIENT_PORT}"
else
  start_ray_worker
fi

MODULE_PATH=${SIMPLE_SWEEP_MODULE_PATH:?missing SIMPLE_SWEEP_MODULE_PATH}
MODULE_ARGS=${SIMPLE_SWEEP_MODULE_ARGS:-}

if [[ "$IS_HEAD" == "true" ]]; then
  echo "[SIMPLE] Launching tool: ${MODULE_PATH} ${MODULE_ARGS}"
  uv run tools/run.py "${MODULE_PATH}" ${MODULE_ARGS}
else
  echo "[SIMPLE] Worker node entering idle loop."
  while true; do sleep 3600; done
fi
