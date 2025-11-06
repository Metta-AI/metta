#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables
EXCLUDE_PATTERN=""

# Note that this script assumes that a .prettierrc is present with appropriate settings for
# markdown files

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier

# Determine mode for final message
if [ "${CHECK_MODE:-false}" = "true" ]; then
  mode_past="checked"
else
  mode_past="formatted"
fi

# Format or check markdown files with Prettier
format_files "md"

echo "All Markdown files (except excluded ones) have been $mode_past with Prettier."
