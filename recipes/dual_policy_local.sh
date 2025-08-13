#!/bin/bash

# Dual-Policy Training Recipe with Checkpoint NPC - Local Version
# This recipe trains a new policy against an old policy checkpoint from WandB locally

set -e

# Configuration
BASE_RUN_NAME="dual_policy_local"
WANDB_CHECKPOINT_URI="${1:-wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2}"
TOTAL_TIMESTEPS="${2:-1000000000}"
NUM_WORKERS="${3:-4}"
GPUS="${4:-4}"
NODES="${5:-8}"
# Rendezvous params (used only when NODES>1); can be overridden via env or CLI positionals
MASTER_ADDR="${6:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${7:-${MASTER_PORT:-29500}}"
NODE_RANK="${8:-${NODE_RANK:-0}}"

# Generate unique run name with counter
TIMESTAMP=$(date +%m-%d)
COUNTER=1
RUN_NAME="${BASE_RUN_NAME}.${TIMESTAMP}.${COUNTER}"

# Find the next available counter
while [ -d "./train_dir/$RUN_NAME" ]; do
    COUNTER=$((COUNTER + 1))
    RUN_NAME="${BASE_RUN_NAME}.${TIMESTAMP}.${COUNTER}"
done

echo "Starting dual-policy training with WandB checkpoint NPC locally"
echo "Run name: $RUN_NAME"
echo "WandB checkpoint URI: $WANDB_CHECKPOINT_URI"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Num workers: $NUM_WORKERS"
echo "GPUs: $GPUS"
echo "Nodes: $NODES"
if [ "$NODES" -gt 1 ]; then
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
fi

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export METTA_RUN_ID="$RUN_NAME"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create run directory
RUN_DIR="./train_dir/$RUN_NAME"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"

# Get local cost estimate (for local runs, this might be 0 or a fixed rate)
export METTA_HOURLY_COST="${METTA_HOURLY_COST:-0.0}"
echo "Estimated hourly cost: $METTA_HOURLY_COST"

# Common training args
TRAIN_ARGS=(
    "run=$RUN_NAME"
    "run_dir=$RUN_DIR"
    "trainer.total_timesteps=$TOTAL_TIMESTEPS"
    "trainer.num_workers=$NUM_WORKERS"
    "trainer.dual_policy.enabled=true"
    "trainer.dual_policy.checkpoint_npc.uri=$WANDB_CHECKPOINT_URI"
    "trainer.dual_policy.training_agents_pct=0.5"
)

# Forward any additional overrides after the first 8 positionals
EXTRA_ARGS=("${@:9}")

# Launch locally with optional multi-GPU / multi-node via torchrun
if [ "$NODES" -eq 1 ] && [ "$GPUS" -eq 1 ]; then
    python tools/train.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
elif [ "$NODES" -eq 1 ]; then
    torchrun --standalone --nproc_per_node="$GPUS" tools/train.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    # Export rendezvous env (can still be overridden by caller)
    export MASTER_ADDR MASTER_PORT NODE_RANK
    torchrun \
        --nproc_per_node="$GPUS" \
        --nnodes="$NODES" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --node_rank="$NODE_RANK" \
        tools/train.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi

echo "Training completed!"
echo "Results available at: $RUN_DIR/"
echo "WandB run: https://wandb.ai/metta-research/dual_policy_training/runs/$RUN_NAME"
echo "Total estimated cost: $METTA_HOURLY_COST per hour"
