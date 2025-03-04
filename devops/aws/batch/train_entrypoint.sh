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

# Update code
git pull

# Install dependencies
pip install -r requirements.txt

# Install specific termcolor version if we checked out a reference
if [ -n "$GIT_REF" ]; then
  pip uninstall -y termcolor
  pip install termcolor==2.4.0
fi

# Setup build
export METTAGRID_REF
./devops/setup_build.sh

# Link training directory
ln -s /mnt/efs/train_dir train_dir 2>/dev/null || true

# Use AWS Batch environment variables for node information
NODE_INDEX=${AWS_BATCH_JOB_NODE_INDEX:-0}
MAIN_NODE_INDEX=${AWS_BATCH_JOB_MAIN_NODE_INDEX:-0}
NUM_NODES=${AWS_BATCH_JOB_NUM_NODES:-1}

# Set up MASTER_ADDR to point to the main node
if [ -f "/etc/hosts.json" ]; then
  # Extract the hostname of the main node from hosts.json
  # The format is typically a JSON array of hostnames
  MASTER_ADDR=$(cat /etc/hosts.json | python3 -c "import sys, json; hosts=json.load(sys.stdin); print(hosts[$MAIN_NODE_INDEX] if $MAIN_NODE_INDEX < len(hosts) else hosts[0])")
  echo "Found hosts.json, using MASTER_ADDR=$MASTER_ADDR"
else
  # Fallback to localhost for single-node jobs
  MASTER_ADDR="localhost"
  echo "No hosts.json found, using MASTER_ADDR=$MASTER_ADDR"
fi

# Export MASTER_ADDR for distributed training
export MASTER_ADDR
export MASTER_PORT=29500

echo "=== Starting training ==="
echo "Run ID: $RUN_ID"
echo "Command: $CMD"
echo "GPUs: $NUM_GPUS"
echo "Node index: $NODE_INDEX of $NUM_NODES nodes"
echo "Main node index: $MAIN_NODE_INDEX"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Workers: $NUM_WORKERS"
echo "Additional args: $TASK_ARGS"

# Run the training command
NUM_GPUS=$NUM_GPUS NUM_NODES=$NUM_NODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ./devops/$CMD.sh run=$RUN_ID hardware=aws trainer.num_workers=$NUM_WORKERS $TASK_ARGS

echo "=== Training complete ==="
