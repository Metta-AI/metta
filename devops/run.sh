#!/bin/bash
set -euo pipefail

detect_visible_gpus() {
  local value="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "$value" ]]; then
    value="${value// /}"
    value="${value%,}"
    if [[ -n "$value" ]]; then
      IFS=',' read -r -a ids <<< "$value"
      echo "${#ids[@]}"
      return
    fi
  fi
  if [[ -n "${RAY_NUM_GPUS:-}" ]]; then
    echo "${RAY_NUM_GPUS}"
    return
  fi
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l
  else
    echo 1
  fi
}

NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}
NODE_INDEX=${NODE_INDEX:-0}

# If Ray assigned GPU IDs but CUDA_VISIBLE_DEVICES is empty, adopt Ray's list.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${RAY_GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RAY_GPU_IDS}"
  echo "[DIAG] Adopted CUDA_VISIBLE_DEVICES from RAY_GPU_IDS: ${CUDA_VISIBLE_DEVICES}"
fi

# Recompute NUM_GPUS if CUDA_VISIBLE_DEVICES now set.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  trimmed="${CUDA_VISIBLE_DEVICES// /}"
  trimmed="${trimmed%,}"
  if [[ -n "$trimmed" ]]; then
    IFS=',' read -r -a __visible_ids <<< "$trimmed"
    NUM_GPUS=${NUM_GPUS:-${#__visible_ids[@]}}
  fi
fi

if [[ -z "${NUM_GPUS:-}" || "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=$(detect_visible_gpus)
fi

# Display configuration
echo "[CONFIG] Training configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Nodes: $NUM_NODES"
echo "  - Master address: $MASTER_ADDR"
echo "  - Master port: $MASTER_PORT"
echo "  - Node index: $NODE_INDEX"
echo "  - Arguments: $*"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export PYTHONOPTIMIZE=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

echo "[INFO] Starting training..."

echo "[DIAG] torch + CUDA visibility before torchrun:"
uv run python <<'PY_EOF' || true
import os
try:
    import torch
except Exception as exc:
    print("[DIAG] torch import failed:", exc)
else:
    print(f"[DIAG] torch.__version__ = {torch.__version__}")
    print(f"[DIAG] torch.version.cuda = {torch.version.cuda!r}")
    print(f"[DIAG] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[DIAG] torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[DIAG] torch.cuda.current_device() = {torch.cuda.current_device()}")
        print(f"[DIAG] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
print(f"[DIAG] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
print(f"[DIAG] LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH')!r}")
print(f"[DIAG] RAY_GPU_IDS = {os.environ.get('RAY_GPU_IDS')!r}")
PY_EOF

# run torchrun; preserve exit code and print a friendly line
set +e
uv run torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$NUM_GPUS \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  --node-rank=$NODE_INDEX \
  tools/run.py \
  "$@"
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[SUCCESS] Training completed successfully"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
exit "$EXIT_CODE"
