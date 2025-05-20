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


# Source environment variables
source ./devops/env.sh

# Link training directory
ln -s /mnt/efs/train_dir train_dir

# Create dist directory
mkdir -p train_dir/dist/$RUN_ID

# Setup log directory and file early
export NODE_INDEX=${AWS_BATCH_JOB_NODE_INDEX:-0}
export LOG_FILE="train_dir/logs/${JOB_NAME}.${NODE_INDEX}.log"

# Create log directory - fail if this doesn't work
mkdir -p $(dirname $LOG_FILE)

# Start logging everything to the log file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Logging to $LOG_FILE ==="

# Function to log timeout-related messages to both stdout and our timeout log
function timeout_log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    # Fail if we can't write to the timeout log
    echo "[$timestamp] TIMEOUT: $message" | tee -a "train_dir/logs/${JOB_NAME}.timeout.log"
}

# Set up the auto-termination feature if JOB_TIMEOUT_MINUTES is set
if [ ! -z "$JOB_TIMEOUT_MINUTES" ] && [ "$NODE_INDEX" = "0" -o "$AWS_BATCH_JOB_NODE_INDEX" = "0" ]; then
    # Create timeout log file - fail if this doesn't work
    mkdir -p "train_dir/logs"
    TIMEOUT_LOG="train_dir/logs/${JOB_NAME}.timeout.log"
    touch $TIMEOUT_LOG
    
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
            
            # Check if the process is still running
            if kill -0 $MAIN_PID 2>/dev/null; then
                timeout_log "Process $MAIN_PID is still running, sending SIGKILL"
                kill -9 $MAIN_PID 2>> $TIMEOUT_LOG || timeout_log "Failed to send SIGKILL to $MAIN_PID"
            else
                timeout_log "Process $MAIN_PID has terminated successfully"
            fi
        else
            timeout_log "Could not identify main process"
            
            # If we can't find the main process, try to find any Python processes
            timeout_log "Attempting to terminate all Python processes"
            pkill -15 python 2>> $TIMEOUT_LOG || timeout_log "No Python processes found to terminate"
            pkill -15 python3 2>> $TIMEOUT_LOG || timeout_log "No Python3 processes found to terminate"
            sleep 5
            pkill -9 python 2>> $TIMEOUT_LOG || true
            pkill -9 python3 2>> $TIMEOUT_LOG || true
        fi
        
        # AWS Batch API call is now a fallback only if process termination fails
        if kill -0 $MAIN_PID 2>/dev/null || pgrep -f python > /dev/null; then
            timeout_log "Process termination failed or other Python processes still running. Trying AWS Batch API as fallback."
            
            # Terminate the job via AWS Batch API
            if [ ! -z "$AWS_BATCH_JOB_ID" ] && command -v aws &> /dev/null && [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
                timeout_log "Terminating AWS Batch job $AWS_BATCH_JOB_ID via AWS API"
                
                # Call AWS Batch API to terminate the job
                aws batch terminate-job \
                    --job-id $AWS_BATCH_JOB_ID \
                    --reason "Timeout of ${TIMEOUT_DISPLAY} reached" \
                    --region us-east-1 2>&1 >> $TIMEOUT_LOG
                
                timeout_log "Termination request sent to AWS Batch API"
            else
                timeout_log "AWS Batch API call not possible (missing AWS_BATCH_JOB_ID, AWS CLI, or credentials)"
            fi
        fi
        
        # Exit the timeout process
        timeout_log "Timeout process complete"
        exit 0
    ) &
    
    # Store the timeout process ID
    TIMEOUT_PID=$!
    timeout_log "Timeout monitor process ID: $TIMEOUT_PID"
    
    # Create a trap to clean up the timeout process if the script exits normally
    trap 'timeout_log "Job completed normally, canceling timeout monitor."; kill $TIMEOUT_PID 2>/dev/null || true' EXIT
    
    # Immediately exit the timeout process if testing
    if [ "$JOB_TIMEOUT_MINUTES" = "test" ]; then
        timeout_log "TEST MODE: Triggering immediate termination"
        kill -9 $TIMEOUT_PID
        aws batch terminate-job \
            --job-id $AWS_BATCH_JOB_ID \
            --reason "Test termination" \
            --region us-east-1
        exit 0
    fi
fi

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

# Install packages and requirements
uv pip install -r requirements.txt
uv pip install -e .
uv --directory mettagrid pip install -e .

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