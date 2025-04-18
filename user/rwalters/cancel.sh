#!/bin/bash
# Script to cancel training jobs started by start_training.sh

# Environment variables file containing the PIDs
PID_FILE="training_pids.env"

# Check if the PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file '$PID_FILE' not found!"
    echo "Make sure you've run start_training.sh first."
    exit 1
fi

# Source the environment variables from the PID file
source "$PID_FILE"

# Check if processes are still running
echo "Checking if training processes are still running..."

CONTROL_RUNNING=0
CURRICULUM_RUNNING=0

if ps -p $CONTROL_PID > /dev/null; then
    CONTROL_RUNNING=1
    echo "Control_${RUN_ID} job (PID: $CONTROL_PID) is running."
else
    echo "Control_${RUN_ID} job (PID: $CONTROL_PID) is not running."
fi

if ps -p $CURRICULUM_PID > /dev/null; then
    CURRICULUM_RUNNING=1
    echo "Curriculum_${RUN_ID} job (PID: $CURRICULUM_PID) is running."
else
    echo "Curriculum_${RUN_ID} job (PID: $CURRICULUM_PID) is not running."
fi

# If both processes are not running, exit
if [ $CONTROL_RUNNING -eq 0 ] && [ $CURRICULUM_RUNNING -eq 0 ]; then
    echo "No training jobs are currently running."
    exit 0
fi

# Ask for confirmation before killing processes
read -p "Do you want to cancel the running training jobs? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Kill the processes if they're running
if [ $CONTROL_RUNNING -eq 1 ]; then
    echo "Canceling control_${RUN_ID} job (PID: $CONTROL_PID)..."
    kill $CONTROL_PID
    if [ $? -eq 0 ]; then
        echo "Control_${RUN_ID} job successfully canceled."
    else
        echo "Failed to cancel control_${RUN_ID} job. Try manually with: kill -9 $CONTROL_PID"
    fi
fi

if [ $CURRICULUM_RUNNING -eq 1 ]; then
    echo "Canceling curriculum_${RUN_ID} job (PID: $CURRICULUM_PID)..."
    kill $CURRICULUM_PID
    if [ $? -eq 0 ]; then
        echo "Curriculum_${RUN_ID} job successfully canceled."
    else
        echo "Failed to cancel curriculum_${RUN_ID} job. Try manually with: kill -9 $CURRICULUM_PID"
    fi
fi

echo "All running training jobs have been canceled."
echo "Log files for this run can be found in: ./training_logs/ with timestamp $TIMESTAMP and run ID $RUN_ID"
