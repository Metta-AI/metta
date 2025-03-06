#!/bin/bash

# AWS Batch Command Line Interface
# Usage: cmd.sh [resource_type] [id] [command] [options]

# Set the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Execute the Python command handler
python3 "$SCRIPT_DIR/cmd.py" "$@"
