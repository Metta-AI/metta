#!/bin/bash

# AWS Batch Command Line Interface
# Usage: cmd.sh [resource_type] [id] [command] [options]

# Set the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v uv &> /dev/null; then
  echo "Error: uv is required but not found"
  exit 1
fi

# Process arguments
NO_COLOR=""
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--no-color" ]; then
    NO_COLOR="--no-color"
  else
    ARGS+=("$arg")
  fi
done

# Check if the first argument is "launch"
if [ "${ARGS[0]}" = "launch" ]; then
  # Remove the "launch" argument
  LAUNCH_ARGS=("${ARGS[@]:1}")

  # Execute the launch_cmd.py script with the remaining arguments
  "$SCRIPT_DIR/batch/launch_cmd.py" $NO_COLOR "${LAUNCH_ARGS[@]}"
else
  # Execute the any other command
  cd "$PROJECT_ROOT"
  uv run -m devops.aws.cmd "$@"
fi
