#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC - Cloud Version
# This recipe trains a new policy against an old policy checkpoint from WandB on cloud infrastructure
#
# Dual policy stats are now properly aggregated across all distributed nodes before logging to wandb,
# so multi-node training (default: 8 nodes) will correctly report all dual policy rewards.

set -e

export METTA_VECENV_RECV_TIMEOUT=300

# Configuration
WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${2:-10000000000}"
NUM_WORKERS="${3:-2}"
GPUS="${4:-4}"
NODES="${5:-8}"
ZERO_COPY="${6:-false}"

# Generate unique run name
RUN_NAME="dual_policy_cloud.$(date +%m-%d).$(date +%H%M).g${GPUS}n${NODES}"

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
  trainer.simulation.skip_git_check=true \
  trainer.dual_policy.enabled=true \
  trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
  trainer.dual_policy.training_agents_pct=0.5 \
  trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  "$@"


echo "Cloud training launched!"
echo "Run name: $RUN_NAME"
echo "Monitor progress at: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"





########################################################################
########################################################################


# #!/bin/bash

# # Dual-Policy Training Recipe with Checkpoint NPC - Local Version
# # This recipe trains a new policy against an old policy checkpoint from WandB locally

# set -e

# # Configuration
# BASE_RUN_NAME="dual_policy_local"
# WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
# TOTAL_TIMESTEPS="${2:-1000000000}"
# NUM_WORKERS="${3:-4}"

# # Generate unique run name with counter
# TIMESTAMP=$(date +%m-%d)
# COUNTER=1
# RUN_NAME="${BASE_RUN_NAME}.${TIMESTAMP}.${COUNTER}"

# # Find the next available counter
# while [ -d "./train_dir/$RUN_NAME" ]; do
#     COUNTER=$((COUNTER + 1))
#     RUN_NAME="${BASE_RUN_NAME}.${TIMESTAMP}.${COUNTER}"
# done

# echo "Starting dual-policy training with WandB checkpoint NPC locally"
# echo "Run name: $RUN_NAME"
# echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
# echo "Total timesteps: $TOTAL_TIMESTEPS"
# echo "Num workers: $NUM_WORKERS"

# # Set up environment
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export METTA_RUN_ID="$RUN_NAME"

# # Activate virtual environment if it exists
# if [ -f ".venv/bin/activate" ]; then
#     source .venv/bin/activate
# fi

# # Create run directory
# RUN_DIR="./train_dir/$RUN_NAME"
# mkdir -p "$RUN_DIR"

# echo "Run directory: $RUN_DIR"

# # Get local cost estimate (for local runs, this might be 0 or a fixed rate)
# export METTA_HOURLY_COST="${METTA_HOURLY_COST:-0.0}"
# echo "Estimated hourly cost: $METTA_HOURLY_COST"

# # Run training using the train.py tool
# python tools/train.py \
#     run="$RUN_NAME" \
#     run_dir="$RUN_DIR" \
#     trainer.total_timesteps="$TOTAL_TIMESTEPS" \
#     trainer.num_workers="$NUM_WORKERS" \
#     trainer.simulation.skip_git_check=true \
#     trainer.dual_policy.enabled=true \
#     trainer.dual_policy.checkpoint_npc.uri="$WANDB_CHECKPOINT_URI" \
#     trainer.dual_policy.training_agents_pct=0.5 \
#     "${@:4}"

# echo "Training completed!"
# echo "Results available at: $RUN_DIR/"
# echo "WandB run: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
# echo "Total estimated cost: $METTA_HOURLY_COST per hour"


