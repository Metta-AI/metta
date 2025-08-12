#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC - Cloud Version
# This recipe trains a new policy against an old policy checkpoint from WandB on cloud infrastructure

set -e

export METTA_VECENV_RECV_TIMEOUT=300

# Configuration
WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${2:-1000000000}"
NUM_WORKERS="${3:-1}"
GPUS="${4:-1}"
NODES="${5:-1}"
ZERO_COPY="${6:-false}"

# Generate unique run name
RUN_NAME="dual_policy_cloud.$(date +%m-%d).$(date +%H%M)"

echo "Starting dual-policy training with WandB checkpoint NPC on cloud"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"
echo "GPUs: $GPUS"
echo "Nodes: $NODES"

export HYDRA_FULL_ERROR=1

# Launch training on cloud using skypilot
./devops/skypilot/launch.py train \
  --gpus="$GPUS" \
  --nodes="$NODES" \
  --no-spot \
  --skip-git-check \
  run="$RUN_NAME" \
  trainer.zero_copy="$ZERO_COPY" \
  trainer.total_timesteps="$TOTAL_TIMESTEPS" \
  trainer.num_workers="$NUM_WORKERS" \
  trainer.dual_policy.enabled=true \
  trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
  trainer.dual_policy.training_agents_pct=0.5 \
  trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  +user=npc_uri_env \
  "$@"


echo "Cloud training launched!"
echo "Run name: $RUN_NAME"
echo "Monitor progress at: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
