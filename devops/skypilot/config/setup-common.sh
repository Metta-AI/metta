#!/bin/bash
set -e
echo "[SETUP] Running common setup..."

cd /workspace/metta

# Initialize git and point to remote
git init
git remote add origin https://github.com/yourusername/metta.git

# Git setup
echo "[SETUP] Git operations..."
git fetch --depth=1000 origin "$METTA_GIT_REF" || git fetch origin
git checkout "$METTA_GIT_REF"

echo "[SETUP] Checked out: $(git rev-parse HEAD)"

# Python environment setup
echo "[SETUP] Setting up Python environment..."
uv sync

# Create required directories
mkdir -p "$WANDB_DIR"

# Setup bash environment
echo "[SETUP] Configuring bash environment..."
cat >> ~/.bashrc << 'EOF'

# Metta environment
cd /workspace/metta
. .venv/bin/activate
. devops/setup.env

# GPU cluster environment variables
export NUM_GPUS=${SKYPILOT_NUM_GPUS_PER_NODE}
export NUM_NODES=${SKYPILOT_NUM_NODES}
export MASTER_ADDR=$(echo "${SKYPILOT_NODE_IPS}" | head -n1)
export MASTER_PORT=8008
export NODE_INDEX=${SKYPILOT_NODE_RANK}
export NCCL_SHM_DISABLE=1
EOF

# Create job secrets
if [ -f ./devops/skypilot/create_job_secrets.py ]; then
    if [ -z "$WANDB_PASSWORD" ]; then
        echo "[SETUP] ERROR: WANDB_PASSWORD environment variable is required but not set"
        echo "[SETUP] Please ensure WANDB_PASSWORD is set in your Skypilot environment variables"
        exit 1
    fi

    echo "[SETUP] Creating job secrets..."

    # Build command - wandb-password is always included
    CMD="./devops/skypilot/create_job_secrets.py --wandb-password \"$WANDB_PASSWORD\""

    # Add observatory-token only if it's set
    if [ -n "$OBSERVATORY_TOKEN" ]; then
        CMD="$CMD --observatory-token \"$OBSERVATORY_TOKEN\""
    fi

    # Execute the command
    eval $CMD || {
        echo "[SETUP] ERROR: Failed to create job secrets"
        exit 1
    }
else
    echo "[SETUP] Warning: create_job_secrets.py not found at ./devops/skypilot/create_job_secrets.py"
    echo "[SETUP] Skipping job secrets creation"
fi

echo "[SETUP] Common setup completed"
