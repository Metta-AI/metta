#!/usr/bin/env bash
set -e

echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Which python: $(which python)"
echo "Python executable: $(python -c 'import sys; print(sys.executable)')"

echo "Configuring runtime environment..."

# look up the file path for storing ENV variables -- N.B. scripts run in isolated
# context and can not directly set ENV in the parent

METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
echo "Persisting env vars into: $METTA_ENV_FILE"

# Write all environment variables using heredoc
cat >> "$METTA_ENV_FILE" << 'EOF'
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1

export WANDB_DIR="./wandb"
export DATA_DIR="${DATA_DIR:-./train_dir}"

export NUM_GPUS="${SKYPILOT_NUM_GPUS_PER_NODE}"
export NUM_NODES="${SKYPILOT_NUM_NODES}"
export MASTER_ADDR="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29501}"
export NODE_INDEX="${SKYPILOT_NODE_RANK}"

# NCCL Configuration

export NCCL_PORT_RANGE="${NCCL_PORT_RANGE:-43000-43063}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp39s0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-2}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"

# Debug
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,IPC
export CUDA_LAUNCH_BLOCKING=1

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

EOF


# Create job secrets (idempotent - overwrites if exists)
if [ -z "$WANDB_PASSWORD" ]; then
    echo "ERROR: WANDB_PASSWORD environment variable is required but not set"
    echo "Please ensure WANDB_PASSWORD is set in your Skypilot environment variables"
    exit 1
fi

echo "Creating/updating job secrets..."

# Build command - wandb-password is always included
CMD="uv run ./devops/skypilot/create_job_secrets.py --wandb-password \"$WANDB_PASSWORD\""

# Add observatory-token only if it's set
if [ -n "$OBSERVATORY_TOKEN" ]; then
    CMD="$CMD --observatory-token \"$OBSERVATORY_TOKEN\""
fi

# Execute the command
eval $CMD || {
    echo "ERROR: Failed to create job secrets"
    exit 1
}

echo "Runtime environment configuration completed"
