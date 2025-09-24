#!/usr/bin/env bash
set -e

echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Which python: $(which python)"
echo "Python executable: $(python -c 'import sys; print(sys.executable)')"

echo "Configuring runtime environment..."

mkdir -p "./wandb"

# look up the file path for storing ENV variables -- N.B. scripts run in isolated
# context and can not directly set ENV in the parent

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
mkdir -p "$(dirname "$METTA_ENV_FILE")"
echo "Persisting env vars into: $METTA_ENV_FILE"

# Track job metadata BEFORE writing other env vars
DATA_DIR="${DATA_DIR:-./train_dir}"
JOB_METADATA_DIR="${DATA_DIR}/.job_metadata/${METTA_RUN_ID}"
mkdir -p "$JOB_METADATA_DIR"

# Files to track
RESTART_COUNT_FILE="$JOB_METADATA_DIR/restart_count"
ACCUMULATED_RUNTIME_FILE="$JOB_METADATA_DIR/accumulated_runtime"
CLUSTER_STOP_FILE="$JOB_METADATA_DIR/cluster_stop"
TERMINATION_REASON_FILE="$JOB_METADATA_DIR/termination_reason"
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$JOB_METADATA_DIR/heartbeat_file}"

# Initialize or update restart tracking
if [ -f "$RESTART_COUNT_FILE" ]; then
  RESTART_COUNT=$(cat "$RESTART_COUNT_FILE")
  RESTART_COUNT=$((${RESTART_COUNT:-0} + 1))
else
  RESTART_COUNT=0
fi

if [[ "$IS_MASTER" == "true" ]]; then
  echo "$RESTART_COUNT" > "$RESTART_COUNT_FILE"
  # Clear any stale cluster stopping state at the beginning of a fresh attempt
  : > "$CLUSTER_STOP_FILE" 2> /dev/null || true
  : > "$TERMINATION_REASON_FILE"
else
  echo "[INFO] Skipping RESTART_COUNT_FILE and CLUSTER_STOP_FILE updates on non-master node"
fi

# Read accumulated runtime
if [ -f "$ACCUMULATED_RUNTIME_FILE" ]; then
  ACCUMULATED_RUNTIME=$(cat "$ACCUMULATED_RUNTIME_FILE")
else
  ACCUMULATED_RUNTIME=0
fi

echo "============= RESTART INFO ============="
echo "  METTA_RUN_ID: ${METTA_RUN_ID}"
echo "  RESTART_COUNT: ${RESTART_COUNT}"
echo "  ACCUMULATED_RUNTIME: ${ACCUMULATED_RUNTIME}s ($((ACCUMULATED_RUNTIME / 60))m)"
echo "  METADATA_DIR: ${JOB_METADATA_DIR}"
echo "========================================"

if [ -f "$METTA_ENV_FILE" ]; then
  echo "Warning: $METTA_ENV_FILE already exists, appending new content"
fi

# Write all environment variables using heredoc
cat >> "$METTA_ENV_FILE" << EOF
export PYTHONUNBUFFERED=1
export PYTHONPATH="\${PYTHONPATH:+\$PYTHONPATH:}\$(pwd)"
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1

export WANDB_DIR="./wandb"
export WANDB_API_KEY="\${WANDB_PASSWORD}"
export DATA_DIR="\${DATA_DIR:-./train_dir}"

# Datadog configuration
export DD_ENV="production"
export DD_SERVICE="skypilot-worker"
export DD_AGENT_HOST="localhost"
export DD_TRACE_AGENT_PORT="8126"

export NUM_GPUS="\${SKYPILOT_NUM_GPUS_PER_NODE}"
export NUM_NODES="\${SKYPILOT_NUM_NODES}"
export MASTER_ADDR="\$(echo "\$SKYPILOT_NODE_IPS" | head -n1)"
export MASTER_PORT="\${MASTER_PORT:-29501}"
export NODE_INDEX="\${SKYPILOT_NODE_RANK}"

# Job metadata exports
export RESTART_COUNT="${RESTART_COUNT}"
export ACCUMULATED_RUNTIME="${ACCUMULATED_RUNTIME}"

# File path exports for monitors
export JOB_METADATA_DIR="${JOB_METADATA_DIR}"
export ACCUMULATED_RUNTIME_FILE="${ACCUMULATED_RUNTIME_FILE}"
export CLUSTER_STOP_FILE="${CLUSTER_STOP_FILE}"
export HEARTBEAT_FILE="${HEARTBEAT_FILE}"
export TERMINATION_REASON_FILE="${TERMINATION_REASON_FILE}"

# NCCL Configuration
export NCCL_PORT_RANGE="\${NCCL_PORT_RANGE:-43000-43063}"
export NCCL_SOCKET_FAMILY="\${NCCL_SOCKET_FAMILY:-AF_INET}"

# Debug
export TORCH_NCCL_ASYNC_ERROR_HANDLING="\${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=""

# NCCL Mode
export NCCL_P2P_DISABLE="\${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="\${NCCL_SHM_DISABLE:-0}"
export NCCL_IB_DISABLE="\${NCCL_IB_DISABLE:-1}"

EOF

# Create job secrets (idempotent - overwrites if exists)
if [ -z "$WANDB_PASSWORD" ]; then
  echo "ERROR: WANDB_PASSWORD environment variable is required but not set"
  echo "Please ensure WANDB_PASSWORD is set in your Skypilot environment variables"
  exit 1
fi

echo "Creating/updating job secrets..."

CMD=(uv run ./devops/skypilot/config/lifecycle/create_job_secrets.py --profile softmax-docker --wandb-password="$WANDB_PASSWORD")

if [ -n "$OBSERVATORY_TOKEN" ]; then
  echo "Found OBSERVATORY_TOKEN and providing to create_job_secrets.py - Observatory features should be available!"
  CMD+=(--observatory-token="$OBSERVATORY_TOKEN")
else
  echo "Warning: OBSERVATORY_TOKEN is not set - Observatory features will not be available."
fi

# Execute without eval
if ! "${CMD[@]}"; then
  echo "ERROR: Failed to create job secrets"
  exit 1
fi

echo "Runtime environment configuration completed"
