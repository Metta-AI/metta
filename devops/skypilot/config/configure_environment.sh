#!/bin/bash
set -e

echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Which python: $(which python)"
echo "Python location: $(python -c 'import sys; print(sys.executable)')"

echo "Configuring runtime environment..."

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG="INFO"
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

# GPU cluster environment variables
export NUM_GPUS=${SKYPILOT_NUM_GPUS_PER_NODE}
export NUM_NODES=${SKYPILOT_NUM_NODES}
export MASTER_ADDR=$(echo "${SKYPILOT_NODE_IPS}" | head -n1)
export MASTER_PORT=8008
export NODE_INDEX=${SKYPILOT_NODE_RANK}
export NCCL_SHM_DISABLE=1

echo "Cluster configuration:"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  NUM_NODES=$NUM_NODES"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  NODE_INDEX=$NODE_INDEX"

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
