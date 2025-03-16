#!/bin/bash
set -e

# Setup script for AWS Batch training jobs

# Environment variables used:
# - RUN_ID: The run ID for the training job
# - CMD: The command to run (train, sweep, evolve)
# - GIT_REF: Git reference (branch or commit) to checkout
# - METTAGRID_REF: Mettagrid reference (branch or commit) to use
# - NUM_GPUS: Number of GPUs per node
# - NUM_WORKERS: Number of workers for training
# - TASK_ARGS: Additional arguments to pass to the training command
#
# AWS Batch environment variables used:
# - AWS_BATCH_JOB_NODE_INDEX: Index of this node in the job
# - AWS_BATCH_JOB_MAIN_NODE_INDEX: Index of the main node
# - AWS_BATCH_JOB_NUM_NODES: Total number of nodes in the job

echo "=== Setting up environment ==="
# Handle git reference if specified
if [ -n "$GIT_REF" ]; then
  echo "Checking out git reference: $GIT_REF"
  git checkout "$GIT_REF"
else
  echo "No git reference specified, using current branch"
fi

# Install dependencies
pip install -r requirements.txt

pip uninstall -y termcolor
pip install termcolor==2.4.0

# Setup build
export METTAGRID_REF
./devops/setup_build.sh

# Link training directory
ln -s /mnt/efs/train_dir train_dir 2>/dev/null || true
# Create dist directory
mkdir -p train_dir/dist/$RUN_ID

export NODE_RANK=${AWS_BATCH_JOB_NODE_INDEX:-0}
export NUM_NODES=${AWS_BATCH_JOB_NUM_NODES:-1}
export MASTER_ADDR=${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-localhost}
export MASTER_PORT=29500
export HARDWARE=${HARDWARE:-aws}
export SKIP_BUILD=1

echo "=== Starting training ==="
echo "Run ID: $RUN_ID"
echo "Command: $CMD"
echo "GPUs: $NUM_GPUS"
echo "Node index: $NODE_RANK of $NUM_NODES nodes"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Workers: $NUM_WORKERS"
echo "Additional args: $TASK_ARGS"
echo "Hardware: $HARDWARE"

# Run the training command
./devops/$CMD.sh run=$RUN_ID hardware=$HARDWARE dist_cfg_path=./train_dir/dist/$RUN_ID/dist_cfg.yaml trainer.num_workers=$NUM_WORKERS $TASK_ARGS

echo "=== Batch job complete ==="
