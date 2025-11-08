#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables
EXCLUDE_PATTERN=""

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier
ensure_prettier_plugin "prettier-plugin-sh" "Prettier plugin for shell scripts"

# Determine mode for final message
if [ "${CHECK_MODE:-false}" = "true" ]; then
  mode_past="checked"
else
  mode_past="formatted"
fi

# Format shell script files
format_files "sh"
