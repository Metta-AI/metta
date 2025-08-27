#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/format_script_tools.sh"

# Initialize variables with default exclusion pattern
EXCLUDE_PATTERN="/configs/\|/scenes/\|/charts/"
EXCLUDE_HELP_TEXT="  Default: excludes paths containing /configs/ or /scenes/ or /charts/
  Use --exclude none to format all files"

# Parse command line arguments
parse_format_args "$@"

# Ensure required tools are available
ensure_pnpm
ensure_prettier

# Show what exclusion pattern is being used
if [ -n "$EXCLUDE_PATTERN" ]; then
  echo "Using exclusion pattern: $EXCLUDE_PATTERN"
else
  echo "No exclusion pattern - formatting all YAML files"
fi
echo ""

# Format YAML files
format_files "yml"
format_files "yaml"

echo ""
echo "All matching YAML files have been formatted with Prettier."
