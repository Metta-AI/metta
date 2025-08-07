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

echo "export PYTHONUNBUFFERED=${1}"                                  >> "$METTA_ENV_FILE"
echo "export PYTHONPATH=\"${PYTHONPATH:+$PYTHONPATH:}$(pwd)\""       >> "$METTA_ENV_FILE"
echo "export PYTHONOPTIMIZE=${1}"                                    >> "$METTA_ENV_FILE"
echo "export HYDRA_FULL_ERROR=${1}"                                  >> "$METTA_ENV_FILE"

# echo "export NCCL_DEBUG=\"INFO\""                                    >> "$METTA_ENV_FILE"
echo "export WANDB_DIR=\"./wandb\""                                  >> "$METTA_ENV_FILE"
echo "export DATA_DIR=\"${DATA_DIR:-./train_dir}\""                  >> "$METTA_ENV_FILE"

echo "export NUM_GPUS=\"${SKYPILOT_NUM_GPUS_PER_NODE}\""              >> "$METTA_ENV_FILE"
echo "export NUM_NODES=\"${SKYPILOT_NUM_NODES}\""                     >> "$METTA_ENV_FILE"
echo "export MASTER_ADDR=\"$(echo "$SKYPILOT_NODE_IPS" | head -n1)\"" >> "$METTA_ENV_FILE"
echo "export MASTER_PORT=\"8008\""                                    >> "$METTA_ENV_FILE"
echo "export NODE_INDEX=\"${SKYPILOT_NODE_RANK}\""                    >> "$METTA_ENV_FILE"

# echo "export NCCL_SHM_DISABLE=${1}"                                   >> "$METTA_ENV_FILE"
echo "export NCCL_DEBUG=\"INFO\""                                    >> "$METTA_ENV_FILE"
echo "export NCCL_DEBUG_SUBSYS=\"ALL\""                              >> "$METTA_ENV_FILE"

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
