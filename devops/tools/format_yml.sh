#!/bin/bash

# Initialize variables with default exclusion pattern
EXCLUDE_PATTERN="/configs/\|/scenes/\|/charts/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exclude)
      if [[ "$2" == "none" ]]; then
        EXCLUDE_PATTERN=""
      else
        EXCLUDE_PATTERN="$2"
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--exclude \"pattern\"]"
      echo "  Default: excludes paths containing /configs/ or /scenes/ or /charts/"
      echo "  Use --exclude none to format all files"
      exit 1
      ;;
  esac
done

# Check if npx and prettier are available
if ! command -v npx &> /dev/null; then
  echo "Error: npx not found. Please install Node.js and npm."
  echo "On Mac, you can install it using:"
  echo "  brew install node"
  echo "Or download from https://nodejs.org/"
  exit 1
fi

# Check if prettier is installed
if ! npx prettier --version &> /dev/null; then
  echo "Prettier not found. Would you like to install it? (y/n)"
  read -r answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Installing prettier..."
    npm install --save-dev prettier
    echo "Prettier installed successfully."
  else
    echo "Prettier is required for this script. Exiting."
    exit 1
  fi
fi

# Function to format files with exclusion
format_files() {
  local file_ext="$1"
  echo "Formatting *.$file_ext files..."

  if [ -n "$EXCLUDE_PATTERN" ]; then
    echo "  Excluding: $EXCLUDE_PATTERN"
    # Use grep -v to exclude files matching the pattern
    find . -name "*.$file_ext" -type f | grep -v "$EXCLUDE_PATTERN" | xargs npx prettier --write
  else
    echo "  Formatting all files (no exclusions)"
    # Format all files if no exclusion pattern
    find . -name "*.$file_ext" -type f | xargs npx prettier --write
  fi
}

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
