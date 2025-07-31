#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC using SkyPilot
# This recipe trains a new policy against an old policy checkpoint from WandB

set -e

# Configuration
RUN_NAME="${1:-$USER.dual_policy_sky.$(date +%m-%d)}"
WANDB_CHECKPOINT_URI="${2:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${3:-10000000000}"
NUM_WORKERS="${4:-4}"

echo "Starting dual-policy training with WandB checkpoint NPC using SkyPilot"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"

# Launch using SkyPilot
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    +user=dual_policy_checkpoint_example \
    run=$RUN_NAME \
    trainer.total_timesteps=$TOTAL_TIMESTEPS \
    trainer.num_workers=$NUM_WORKERS \
    trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
    "$@"

sky status

echo "Results available at: ./train_dir/$RUN_NAME/"
echo "WandB run: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
