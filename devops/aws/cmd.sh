#!/bin/bash

# AWS Batch Command Line Interface
# Usage: cmd.sh [resource_type] [id] [command] [options]

# Set the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Check if the first argument is "launch"
if [ "$1" = "launch" ]; then
    # Shift the first argument to remove "launch"
    shift

    # Execute the launch_cmd.py script with the remaining arguments
    PYTHONPATH="$PROJECT_ROOT" python3 "$SCRIPT_DIR/batch/launch_cmd.py" "$@"
else
    # Execute the Python command handler as a module
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python3 -m devops.aws.cmd "$@"
fi
