#!/bin/bash
# Initialize variables
EXCLUDE_PATTERN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exclude)
      EXCLUDE_PATTERN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--exclude \"pattern\"]"
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
    # Use grep -v to exclude files matching the pattern
    find . -name "*.$file_ext" -type f | grep -v "$EXCLUDE_PATTERN" | xargs npx prettier --write
  else
    # Format all files if no exclusion pattern
    find . -name "*.$file_ext" -type f | xargs npx prettier --write
  fi
}

# Format JSON files
format_files "json"

# Also format .code-workspace files
if [ -f "metta.code-workspace" ]; then
  echo "Formatting metta.code-workspace..."
  npx prettier --write metta.code-workspace
fi

# Check if prettier config exists
if [ ! -f ".prettierrc" ] && [ ! -f ".prettierrc.json" ] && [ ! -f ".prettierrc.js" ]; then
  echo ""
  echo "WARNING: No .prettierrc file found in the current directory."
  echo "The JSON5 parser configuration for .vscode/*.json files won't be applied."
  echo "Make sure your .prettierrc file is in the project root."
fi

echo "All JSON files (except excluded ones) have been formatted with Prettier."