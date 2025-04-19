#!/bin/bash
# Script to cancel training jobs started by start_training.sh
# Uses killall to find and kill all Python processes
echo "WARNING: This script will kill all Python processes"
echo "This may affect other Python processes if they are running."
# Show which processes will be killed
echo "The following processes will be terminated:"
ps aux | grep "python" | grep -v grep
# Final confirmation after seeing the process list
read -p "Confirm termination of these processes? (y/n): " final_confirm
if [[ $final_confirm != [yY] && $final_confirm != [yY][eE][sS] ]]; then
echo "Operation cancelled."
exit 0
fi
# Kill the processes
echo "Canceling all training jobs..."
pkill -f "python"
# Check if the kill was successful
if [ $? -eq 0 ]; then
echo "Training jobs successfully canceled."
else
echo "No matching processes found or error occurred."
fi
echo "To view logs, check the ./training_logs/ directory."