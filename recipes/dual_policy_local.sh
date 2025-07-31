#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC - Local Version
# This recipe trains a new policy against an old policy checkpoint from WandB locally

set -e

# Configuration
RUN_NAME="${1:-$USER.dual_policy_local.$(date +%m-%d)}"
WANDB_CHECKPOINT_URI="${2:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${3:-1000000000}"
NUM_WORKERS="${4:-4}"

echo "Starting dual-policy training with WandB checkpoint NPC locally"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export METTA_RUN_ID="$RUN_NAME"

# Create run directory
RUN_DIR="./train_dir/$RUN_NAME"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"

# Run training locally
python run.py \
    run="$RUN_NAME" \
    trainer.total_timesteps="$TOTAL_TIMESTEPS" \
    trainer.num_workers="$NUM_WORKERS" \
    trainer.dual_policy.enabled=true \
    trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
    "$@"

echo "Training completed!"
echo "Results available at: $RUN_DIR/"
echo "WandB run: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
