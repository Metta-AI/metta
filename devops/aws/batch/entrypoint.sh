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
# - JOB_TIMEOUT_MINUTES: Optional timeout in minutes for auto-termination
#
# AWS Batch environment variables used:
# - AWS_BATCH_JOB_ID: The ID of the current job
# - AWS_BATCH_JOB_NODE_INDEX: Index of this node in the job
# - AWS_BATCH_JOB_MAIN_NODE_INDEX: Index of the main node
# - AWS_BATCH_JOB_NUM_NODES: Total number of nodes in the job

# Set up the auto-termination feature if JOB_TIMEOUT_MINUTES is set
if [ ! -z "$JOB_TIMEOUT_MINUTES" ] && [ "$NODE_INDEX" = "0" -o "$AWS_BATCH_JOB_NODE_INDEX" = "0" ]; then
    # Only set up timeout on the main node
    TIMEOUT_SECONDS=$((JOB_TIMEOUT_MINUTES * 60))
    
    # Format timeout for display
    if [ $JOB_TIMEOUT_MINUTES -ge 60 ]; then
        HOURS=$((JOB_TIMEOUT_MINUTES / 60))
        MINS=$((JOB_TIMEOUT_MINUTES % 60))
        if [ $MINS -eq 0 ]; then
            TIMEOUT_DISPLAY="${HOURS}h"
        else
            TIMEOUT_DISPLAY="${HOURS}h ${MINS}m"
        fi
    else
        TIMEOUT_DISPLAY="${JOB_TIMEOUT_MINUTES}m"
    fi
    
    echo "AUTO-TERMINATION: Job will terminate after ${TIMEOUT_DISPLAY}"
    
    # Start the timeout monitor in the background
    (
        # Sleep for the specified timeout duration
        sleep $TIMEOUT_SECONDS
        
        echo "JOB_TIMEOUT_MINUTES (${TIMEOUT_DISPLAY}) reached. Initiating job termination..."
        
        # Record timeout in a specific log file
        TIMEOUT_LOG="train_dir/logs/${JOB_NAME}.timeout.log"
        echo "$(date): JOB TERMINATED DUE TO TIMEOUT AFTER ${TIMEOUT_DISPLAY}" > $TIMEOUT_LOG
        
        # Try to get the main training process to allow it to save checkpoints
        MAIN_PID=$(ps -eo pid,%cpu --sort=-%cpu | awk 'NR==2 {print $1}')
        
        if [ ! -z "$MAIN_PID" ]; then
            echo "Sending SIGTERM to main process $MAIN_PID to allow checkpoint saving" | tee -a $TIMEOUT_LOG
            kill -15 $MAIN_PID || true
            
            # Give it some time to save checkpoints
            echo "Waiting 60 seconds for graceful termination and checkpoint saving..." | tee -a $TIMEOUT_LOG
            sleep 60
        fi
        
        # Terminate the job via AWS Batch API
        if [ ! -z "$AWS_BATCH_JOB_ID" ]; then
            echo "Terminating AWS Batch job $AWS_BATCH_JOB_ID via AWS API" | tee -a $TIMEOUT_LOG
            aws batch terminate-job \
                --job-id $AWS_BATCH_JOB_ID \
                --reason "Timeout of ${TIMEOUT_DISPLAY} reached" \
                --region us-east-1
            echo "Termination request sent to AWS Batch API" | tee -a $TIMEOUT_LOG
        else
            echo "ERROR: AWS_BATCH_JOB_ID not found. Cannot terminate job via API." | tee -a $TIMEOUT_LOG
        fi
        
        # Wait for AWS Batch to process the termination
        echo "Waiting for AWS Batch to process the termination request..." | tee -a $TIMEOUT_LOG
        sleep 30
        
        # As a last resort, attempt to shut down the container
        echo "Forcing exit of the entrypoint script" | tee -a $TIMEOUT_LOG
        exit 1
    ) &
    
    # Store the timeout process ID
    TIMEOUT_PID=$!
    
    # Create a trap to clean up the timeout process if the script exits normally
    trap 'echo "Job completed normally, canceling timeout monitor."; kill $TIMEOUT_PID 2>/dev/null || true' EXIT
fi

# Link training directory
ln -s /mnt/efs/train_dir train_dir 2> /dev/null || true
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

# Log timeout information if set
if [ ! -z "$JOB_TIMEOUT_MINUTES" ]; then
    if [ $JOB_TIMEOUT_MINUTES -ge 60 ]; then
        HOURS=$((JOB_TIMEOUT_MINUTES / 60))
        MINS=$((JOB_TIMEOUT_MINUTES % 60))
        if [ $MINS -eq 0 ]; then
            TIMEOUT_DISPLAY="${HOURS}h"
        else
            TIMEOUT_DISPLAY="${HOURS}h ${MINS}m"
        fi
    else
        TIMEOUT_DISPLAY="${JOB_TIMEOUT_MINUTES}m"
    fi
    
    echo "=== AUTO-TERMINATION: Job will terminate after ${TIMEOUT_DISPLAY} ==="
fi

echo "=== Setting up environment ==="
# Handle git reference if specified
if [ -n "$GIT_REF" ]; then
  echo "Checking out git reference: $GIT_REF"
  git checkout "$GIT_REF"
else
  echo "No git reference specified, using current branch"
fi

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
if [ ! -z "$JOB_TIMEOUT_MINUTES" ]; then
    echo "Auto-termination: After ${TIMEOUT_DISPLAY}"
else
    echo "Auto-termination: Not set"
fi
echo "Additional args: $TASK_ARGS"

# Run the training command
./devops/$CMD.sh run=$RUN_ID +hardware=$HARDWARE trainer.num_workers=$NUM_WORKERS $TASK_ARGS

echo "=== Batch job complete ==="
