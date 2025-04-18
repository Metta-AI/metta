#!/bin/bash
# Script to start multiple training jobs in parallel with configurable run ID

# Example Usage:

# ./start_training.sh          # Use default run ID "0"
# ./start_training.sh 1        # Use run ID "1"
# ./start_training.sh test     # Use run ID "test"

# Default run ID (can be overridden)
RUN_ID=${1:-"0"}

echo "Using run ID: $RUN_ID"

# Create a log directory if it doesn't exist
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Store PIDs in environment variables file for later use
PID_FILE="training_pids.env"

# debug
# python -m tools.train +hardware=pufferbox run=rwalters.test trainer.env=/env/mettagrid/robb_map 

# First training job - control experiment
echo "Starting control_$RUN_ID training job..."
python -m tools.train +hardware=pufferbox run=rwalters.control_$RUN_ID \
  trainer.env=/env/mettagrid/robb_map \
  +trainer.env_overrides.game.max_size=60 > "$LOG_DIR/control_${RUN_ID}_$TIMESTAMP.log" 2>&1 &

# Save the process ID for the first job
CONTROL_PID=$!
echo "Control job started with PID: $CONTROL_PID"

# Second training job - curriculum experiment
echo "Starting curriculum_$RUN_ID training job..."
python -m tools.train +hardware=pufferbox run=rwalters.curriculum_$RUN_ID \
  trainer.env=/env/mettagrid/robb_map > "$LOG_DIR/curriculum_${RUN_ID}_$TIMESTAMP.log" 2>&1 &

# Save the process ID for the second job
CURRICULUM_PID=$!
echo "Curriculum job started with PID: $CURRICULUM_PID"

# Write PIDs to environment variables file
echo "CONTROL_PID=$CONTROL_PID" > $PID_FILE
echo "CURRICULUM_PID=$CURRICULUM_PID" >> $PID_FILE
echo "TIMESTAMP=$TIMESTAMP" >> $PID_FILE
echo "RUN_ID=$RUN_ID" >> $PID_FILE

echo "All training jobs have been started in the background."
echo "Process IDs have been saved to $PID_FILE"
echo "Logs are being saved to the $LOG_DIR directory"
echo ""
echo "To check status of these processes, run:"
echo "  ps -p $CONTROL_PID,$CURRICULUM_PID"
echo ""
echo "To cancel these jobs, run the cancel.sh script"
