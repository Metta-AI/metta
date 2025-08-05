#!/bin/bash
set -e
echo "Configuring runtime environment..."

# Python environment setup
echo "Setting up Python environment..."
uv sync

# Create required directories
mkdir -p "$WANDB_DIR"

# Setup bash environment (idempotent)
echo "Configuring bash environment..."

# Check if already configured to avoid duplicate entries
if ! grep -q "# Metta environment" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# Metta environment
cd /workspace/metta
. .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR="./wandb"
export DATA_DIR=${DATA_DIR:-./train_dir}

# GPU cluster environment variables
export NUM_GPUS=${SKYPILOT_NUM_GPUS_PER_NODE}
export NUM_NODES=${SKYPILOT_NUM_NODES}
export MASTER_ADDR=$(echo "${SKYPILOT_NODE_IPS}" | head -n1)
export MASTER_PORT=8008
export NODE_INDEX=${SKYPILOT_NODE_RANK}
export NCCL_SHM_DISABLE=1
EOF
    echo "Bash environment configured"
else
    echo "Bash environment already configured, skipping"
fi

# Create job secrets (idempotent - overwrites if exists)
if [ -z "$WANDB_PASSWORD" ]; then
    echo "ERROR: WANDB_PASSWORD environment variable is required but not set"
    echo "Please ensure WANDB_PASSWORD is set in your Skypilot environment variables"
    exit 1
fi

echo "Creating/updating job secrets..."

# Build command - wandb-password is always included
CMD="./devops/skypilot/create_job_secrets.py --wandb-password \"$WANDB_PASSWORD\""

# Add observatory-token only if it's set
if [ -n "$OBSERVATORY_TOKEN" ]; then
    CMD="$CMD --observatory-token \"$OBSERVATORY_TOKEN\""
fi

# Execute the command
eval $CMD || {
    echo "ERROR: Failed to create job secrets"
    exit 1
}

source ~/.bashrc

echo "Runtime environment configuration completed"
