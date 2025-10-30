#!/usr/bin/env bash

set -euo pipefail

cd /workspace/metta

# Ensure we are using the project virtualenv
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2> /dev/null || true
fi
. .venv/bin/activate

export WRAPPER_PID=$BASHPID

echo "[SIMPLE] Configuring environment..."
bash ./devops/skypilot/config/lifecycle/configure_environment.sh

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
source "$METTA_ENV_FILE"

echo "[SIMPLE] Running CUDA diagnostics..."
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "[SIMPLE] nvidia-smi command failed (above)."
  fi
else
  echo "[SIMPLE] nvidia-smi not found on PATH."
fi

python <<'PY_EOF' || true
import os
try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostic output
    print("[SIMPLE] torch import failed:", exc)
else:
    print(f"[SIMPLE] torch.__version__ = {torch.__version__}")
    print(f"[SIMPLE] torch.version.cuda = {torch.version.cuda!r}")
    print(f"[SIMPLE] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[SIMPLE] torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[SIMPLE] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
print(f"[SIMPLE] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
print(f"[SIMPLE] LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH')!r}")
PY_EOF

export RANK=${SKYPILOT_NODE_RANK:-0}
export IS_HEAD=$([[ "$RANK" == "0" ]] && echo "true" || echo "false")
HEAD_IP=$(echo "${SKYPILOT_NODE_IPS:-127.0.0.1}" | head -n1)
RAY_PORT=${RAY_HEAD_PORT:-6379}
RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-10001}

# Detect GPUs and CPUs on this node (not provided by SkyPilot)
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
else
  GPU_COUNT=0
fi

if command -v nproc >/dev/null 2>&1; then
  CPU_COUNT=$(nproc)
else
  CPU_COUNT=$(python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "[SIMPLE] Warning: No GPUs detected, defaulting Ray to CPU-only mode."
fi

echo "[SIMPLE] Node rank: $RANK (head: $IS_HEAD)"
echo "[SIMPLE] Head IP: $HEAD_IP"
echo "[SIMPLE] GPUs per node: $GPU_COUNT"
echo "[SIMPLE] CPUs per node: $CPU_COUNT"
echo "[SIMPLE] Ray ports: tcp=$RAY_PORT client=$RAY_CLIENT_PORT"

# Export detected hardware counts for Python processes to use
# These are detected by us, not provided by SkyPilot
export METTA_DETECTED_GPUS_PER_NODE="$GPU_COUNT"
export METTA_DETECTED_CPUS_PER_NODE="$CPU_COUNT"

start_ray_head() {
  echo "[SIMPLE] Starting Ray head node..."
  # Let Ray auto-detect CPUs instead of forcing a potentially wrong value
  # Only specify GPUs since we need to ensure they're properly allocated
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
  echo "[SIMPLE] Starting Ray worker..."
  # Let Ray auto-detect CPUs on worker nodes too
  until ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats --num-gpus "$GPU_COUNT"; do
    echo "[SIMPLE] Worker waiting for head..."
    sleep 5
  done
  sleep 5
}

if [[ "$IS_HEAD" == "true" ]]; then
  start_ray_head
  # Use local mode address instead of client mode for better GPU allocation
  # This format allows Ray to properly allocate GPUs to worker processes
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  echo "[SIMPLE] Ray address (local mode): ${RAY_ADDRESS}"
  echo "[SIMPLE] Note: Using local mode for proper GPU allocation to trials"
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
