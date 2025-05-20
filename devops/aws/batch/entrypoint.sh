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

# Source environment variables first to set up any paths
source ./devops/env.sh 2>/dev/null || true

# Setup log directory and file early
export NODE_INDEX=${AWS_BATCH_JOB_NODE_INDEX:-0}
export LOG_FILE="train_dir/logs/${JOB_NAME}.${NODE_INDEX}.log"
mkdir -p $(dirname $LOG_FILE) 2>/dev/null || true

# Function to log timeout-related messages to both stdout and our timeout log
function timeout_log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] TIMEOUT: $message" | tee -a "train_dir/logs/${JOB_NAME}.timeout.log" 2>/dev/null || echo "[$timestamp] TIMEOUT: $message"
}

# Set up the auto-termination feature if JOB_TIMEOUT_MINUTES is set
if [ ! -z "$JOB_TIMEOUT_MINUTES" ] && [ "$NODE_INDEX" = "0" -o "$AWS_BATCH_JOB_NODE_INDEX" = "0" ]; then
    # Create timeout log file
    mkdir -p "train_dir/logs" 2>/dev/null || true
    TIMEOUT_LOG="train_dir/logs/${JOB_NAME}.timeout.log"
    touch $TIMEOUT_LOG 2>/dev/null || TIMEOUT_LOG="/tmp/${JOB_NAME}.timeout.log"
    
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
    
    timeout_log "Setting up auto-termination after ${TIMEOUT_DISPLAY} (${TIMEOUT_SECONDS} seconds)"
    timeout_log "AWS_BATCH_JOB_ID=${AWS_BATCH_JOB_ID}"
    timeout_log "AWS credentials check: $(aws sts get-caller-identity --query 'Account' --output text 2>/dev/null || echo 'NOT AVAILABLE')"
    
    # Start the timeout monitor in the background
    (
        timeout_log "Timeout monitor started. Will wake up after ${TIMEOUT_DISPLAY}"
        
        # Record the start time
        START_TIME=$(date +%s)
        
        # Sleep for the specified timeout duration
        sleep $TIMEOUT_SECONDS
        
        # Calculate how long we actually slept
        END_TIME=$(date +%s)
        ELAPSED_SECONDS=$((END_TIME - START_TIME))
        ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
        
        timeout_log "Waking up after ${ELAPSED_MINUTES}m (${ELAPSED_SECONDS}s). Should have been ${JOB_TIMEOUT_MINUTES}m (${TIMEOUT_SECONDS}s)"
        timeout_log "JOB_TIMEOUT_MINUTES (${TIMEOUT_DISPLAY}) reached. Initiating job termination..."
        
        # Try to get the main training process to allow it to save checkpoints
        timeout_log "Finding main process to terminate gracefully..."
        PS_OUTPUT=$(ps -eo pid,%cpu,command --sort=-%cpu)
        echo "$PS_OUTPUT" | head -n 10 >> $TIMEOUT_LOG
        MAIN_PID=$(echo "$PS_OUTPUT" | awk 'NR==2 {print $1}')
        
        if [ ! -z "$MAIN_PID" ]; then
            timeout_log "Sending SIGTERM to main process $MAIN_PID to allow checkpoint saving"
            kill -15 $MAIN_PID 2>> $TIMEOUT_LOG || timeout_log "Failed to send SIGTERM to $MAIN_PID"
            
            # Give it some time to save checkpoints
            timeout_log "Waiting 30 seconds for graceful termination and checkpoint saving..."
            sleep 30
        else
            timeout_log "Could not identify main process"
        fi
        
        # Terminate the job via AWS Batch API
        if [ ! -z "$AWS_BATCH_JOB_ID" ]; then
            timeout_log "Terminating AWS Batch job $AWS_BATCH_JOB_ID via AWS API"
            
            # Verify AWS CLI is available
            if ! command -v aws &> /dev/null; then
                timeout_log "ERROR: AWS CLI is not installed or not in PATH"
            else
                timeout_log "AWS CLI is available"
                
                # Verify AWS credentials
                if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
                    timeout_log "ERROR: AWS credentials not set in environment"
                else
                    timeout_log "AWS credentials are set"
                    
                    # Call AWS Batch API to terminate the job
                    TERMINATION_RESULT=$(aws batch terminate-job \
                        --job-id $AWS_BATCH_JOB_ID \
                        --reason "Timeout of ${TIMEOUT_DISPLAY} reached" \
                        --region us-east-1 2>&1)
                    
                    TERMINATION_EXIT_CODE=$?
                    if [ $TERMINATION_EXIT_CODE -eq 0 ]; then
                        timeout_log "Successfully sent termination request to AWS Batch API"
                    else
                        timeout_log "ERROR: Failed to terminate job via API: $TERMINATION_EXIT_CODE"
                        timeout_log "API response: $TERMINATION_RESULT"
                    fi
                fi
            fi
        else
            timeout_log "ERROR: AWS_BATCH_JOB_ID not found. Cannot terminate job via API."
        fi
        
        # Wait a bit more, then try more aggressive termination
        timeout_log "Waiting 30 more seconds for API termination to take effect..."
        sleep 30
        
        # If we're still running, try killing all Python processes
        timeout_log "Still running after API termination attempt. Trying to kill all Python processes..."
        pkill -9 python 2>> $TIMEOUT_LOG || timeout_log "No Python processes found to kill"
        pkill -9 python3 2>> $TIMEOUT_LOG || timeout_log "No Python3 processes found to kill"
        
        # As a last resort, exit the script with an error code
        timeout_log "Forcing exit of the script with exit code 143 (SIGTERM)"
        kill -9 -1 2>/dev/null || timeout_log "Failed to send SIGKILL to all processes in the group"
        exit 143
    ) &
    
    # Store the timeout process ID
    TIMEOUT_PID=$!
    timeout_log "Timeout monitor process ID: $TIMEOUT_PID"
    
    # Create a trap to clean up the timeout process if the script exits normally
    trap 'timeout_log "Job completed normally, canceling timeout monitor."; kill $TIMEOUT_PID 2>/dev/null || true' EXIT
fi

# Link training directory
ln -s /mnt/efs/train_dir train_dir 2> /dev/null || true
# Create dist directory
mkdir -p train_dir/dist/$RUN_ID 2>/dev/null || true

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
    echo "=== Timeout process ID: $TIMEOUT_PID ==="
    
    # Print AWS identity if available (to verify credentials)
    if command -v aws &> /dev/null; then
        echo "=== AWS Identity: $(aws sts get-caller-identity --query 'Account' --output text 2>/dev/null || echo 'NOT AVAILABLE') ==="
    else
        echo "=== AWS CLI not available ==="
    fi
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
echo "AWS_BATCH_JOB_ID: $AWS_BATCH_JOB_ID"

# Run the training command
./devops/$CMD.sh run=$RUN_ID +hardware=$HARDWARE trainer.num_workers=$NUM_WORKERS $TASK_ARGS

echo "=== Batch job complete ==="
