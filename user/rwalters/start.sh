#!/bin/bash
# Script to start multiple training jobs in parallel with configurable run ID and verbose option
# Example Usage:
# ./start_training.sh # Use default run ID "0", logs to files only
# ./start_training.sh 1 # Use run ID "1", logs to files only
# ./start_training.sh test # Use run ID "test", logs to files only
# ./start_training.sh 0 verbose # Use run ID "0" with verbose output
# ./start_training.sh test verbose # Use run ID "test" with verbose output

# Default run ID (can be overridden)
RUN_ID=${1:-"0"}
# Check for verbose flag as second argument
VERBOSE=${2:-""}

echo "Using run ID: $RUN_ID"
if [[ "$VERBOSE" == "verbose" ]]; then
  echo "Verbose mode enabled: logs will be shown in terminal and saved to files"
fi

# Create a log directory if it doesn't exist
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to start a job with or without verbose logging
start_job() {
  local job_name="$1"
  local job_cmd="$2"
  local log_file="$LOG_DIR/${job_name}_${RUN_ID}_$TIMESTAMP.log"
  
  echo "Starting ${job_name}_$RUN_ID training job..."
  
  if [[ "$VERBOSE" == "verbose" ]]; then
    # Run command with output to both console and log file
    $job_cmd | tee "$log_file" &
  else
    # Run command with output only to log file
    $job_cmd > "$log_file" 2>&1 &
  fi
}

# First training job - control experiment
CONTROL_CMD="python -m tools.train +hardware=pufferbox run=rwalters.control_$RUN_ID trainer.env=/env/mettagrid/robb_map +trainer.env_overrides.game.difficulty=10"
start_job "control" "$CONTROL_CMD"

# Second training job - curriculum experiment
CURRICULUM_CMD="python -m tools.train +hardware=pufferbox run=rwalters.curriculum_$RUN_ID trainer.env=/env/mettagrid/robb_map"
start_job "curriculum" "$CURRICULUM_CMD"

echo "All training jobs have been started in the background."
echo "Logs are being saved to the $LOG_DIR directory"
echo ""
echo "Note: Due to subprocess handling, PIDs are not tracked in this version."
echo "To monitor or cancel jobs, you may need to use commands like 'ps aux | grep tools.train'"
echo ""
echo "To cancel all python tasks, run the cancel.sh script"
