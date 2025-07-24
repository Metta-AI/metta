#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC
# This recipe trains a new policy against an old policy checkpoint from WandB

set -e

# Configuration
RUN_NAME="dual_policy_vs_checkpoint_$(date +%Y%m%d_%H%M%S)"
WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/metta/model/yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18:latest}"
TOTAL_TIMESTEPS="${2:-10000000000}"
NUM_WORKERS="${3:-4}"

echo "Starting dual-policy training with WandB checkpoint NPC"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"

# Set cost tracking (adjust hourly cost as needed)
export METTA_HOURLY_COST=10.0

# Run the training
python tools/train.py \
    +user=dual_policy_checkpoint_example \
    run=$RUN_NAME \
    trainer.total_timesteps=$TOTAL_TIMESTEPS \
    trainer.num_workers=$NUM_WORKERS \
    trainer.dual_policy.checkpoint_npc.checkpoint_path="$WANDB_CHECKPOINT_URI"

echo "Training completed!"
echo "Results available at: ./train_dir/$RUN_NAME/"
echo "WandB run: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
