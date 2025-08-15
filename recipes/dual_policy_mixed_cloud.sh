#!/bin/bash

# Dual-Policy Training Recipe with Mixed Reward Strategy - Cloud Version
# This recipe trains a new policy with a mixed facilitator/original reward (coef=0.75) on cloud infrastructure

set -e

# Configuration
WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${2:-1000000000}"
NUM_WORKERS="${3:-4}"

# Generate unique run name
RUN_NAME="dual_policy_mixed_cloud.$(date +%m-%d).$(date +%H%M)"

echo "Starting dual-policy training with mixed reward strategy (coef=0.75) on cloud"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"

# Launch training on cloud using skypilot
./devops/skypilot/launch.py train \
  --gpus=1 \
  --nodes=1 \
  --no-spot \
  run="$RUN_NAME" \
  trainer.total_timesteps="$TOTAL_TIMESTEPS" \
  trainer.num_workers="$NUM_WORKERS" \
  trainer.vectorization=multiprocessing \
  trainer.dual_policy.enabled=true \
  trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
  trainer.dual_policy.training_agents_pct=0.5 \
  reward_strategy_name=mixed \
  facilitator_mix_coef=0.75 \
  +user=npc_uri_env \
  "$@"

echo "Cloud training launched!"
echo "Run name: $RUN_NAME"
echo "Monitor progress at: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME" 