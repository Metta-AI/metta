#!/bin/bash
set -e

# Setup script for AWS Batch training jobs

# Environment variables used:
# - RUN_ID: The run ID for the training job
# - CMD: The command to run (train, sweep, evolve)
# - GIT_REF: Git reference (branch or commit) to checkout
# - NUM_GPUS: Number of GPUs per node
# - NUM_WORKERS: Number of workers for training
# - TASK_ARGS: Additional arguments to pass to the training command
#
# AWS Batch environment variables used:
# - AWS_BATCH_JOB_NODE_INDEX: Index of this node in the job
# - AWS_BATCH_JOB_MAIN_NODE_INDEX: Index of the main node
# - AWS_BATCH_JOB_NUM_NODES: Total number of nodes in the job

# Link training directory
ln -s /mnt/efs/train_dir train_dir 2>/dev/null || true
# Create dist directory
mkdir -p train_dir/dist/$RUN_ID

# Source environment variables
source ./devops/env.sh

# Setup log directory and file
export NODE_INDEX=${AWS_BATCH_JOB_NODE_INDEX:-0}
export LOG_FILE="train_dir/logs/${JOB_NAME}.${NODE_INDEX}.log"
mkdir -p $(dirname $LOG_FILE)

# Start logging everything to the log file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Logging to $LOG_FILE ==="

echo "=== Setting up environment ==="
# Handle git reference if specified
if [ -n "$GIT_REF" ]; then
  echo "Checking out git reference: $GIT_REF"
  git checkout "$GIT_REF"
else
  echo "No git reference specified, using current branch"
fi

pip uninstall -y termcolor
pip install termcolor==2.4.0

# Setup build (installs requirements)
./devops/setup_build.sh

export NUM_NODES=${AWS_BATCH_JOB_NUM_NODES:-1}
export MASTER_ADDR=${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export HARDWARE=${HARDWARE:-aws}
export SKIP_BUILD=1
export DIST_ID=$JOB_NAME

# Set up WandB directory
export WANDB_DIR="./wandb"
mkdir -p $WANDB_DIR

echo "=== Starting training ==="
echo "Run ID: $RUN_ID"
echo "Command: $CMD"
echo "GPUs: $NUM_GPUS"
echo "Node index: $NODE_INDEX of $NUM_NODES nodes"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Workers: $NUM_WORKERS"
echo "Hardware: $HARDWARE"
echo "Additional args: $TASK_ARGS"

# Run the training command
./devops/$CMD.sh run=$RUN_ID +hardware=$HARDWARE trainer.num_workers=$NUM_WORKERS $TASK_ARGS

echo "=== Batch job complete ===" 
