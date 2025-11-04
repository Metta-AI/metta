#!/bin/bash
# Formatting-related helper functions

# Default to formatting mode; can be switched to check mode via parse_format_args
FORMAT_MODE="write"

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

# Check if pnpm is available, exit if not
ensure_pnpm() {
  if ! check_cmd pnpm; then
    echo "Error: pnpm not found. Please install pnpm."
    echo "You can install it using one of these methods:"
    echo "  - npm: npm install -g pnpm"
    echo "  - Homebrew: brew install pnpm"
    echo "  - Standalone: curl -fsSL https://get.pnpm.io/install.sh | sh -"
    echo "  - Or visit: https://pnpm.io/installation"
    exit 1
  fi
}

# Check if prettier is installed, offer to install if not
ensure_prettier() {
  if ! pnpm exec prettier --version &> /dev/null; then
    echo "Prettier not found. Would you like to install it? (y/n)"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      echo "Installing prettier..."
      pnpm add --save-dev prettier
      echo "Prettier installed successfully."
    else
      echo "Prettier is required for this script. Exiting."
      exit 1
    fi
  fi
}

# Check if a prettier plugin is installed, offer to install if not
ensure_prettier_plugin() {
  local plugin_name="$1"
  local plugin_description="${2:-$plugin_name}"

  if ! pnpm list "$plugin_name" &> /dev/null; then
    echo "$plugin_description not found. Would you like to install it? (y/n)"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      echo "Installing $plugin_name..."
      pnpm add --save-dev "$plugin_name"
      echo "$plugin_description installed successfully."
    else
      echo "$plugin_description is required for this script. Exiting."
      exit 1
    fi
  fi
}

# Function to format files with exclusion pattern
format_files() {
  local file_ext="$1"
  local exclude_pattern="${2:-$EXCLUDE_PATTERN}"
  local mode="${FORMAT_MODE:-write}"
  local action
  local prettier_flag

  if [ "$mode" = "check" ]; then
    action="Checking formatting for"
    prettier_flag="--check"
  else
    action="Formatting"
    prettier_flag="--write"
  fi

  echo "$action *.$file_ext files..."

  if [ -n "$exclude_pattern" ]; then
    echo "  Excluding: $exclude_pattern"
    if [ "$mode" = "check" ]; then
      echo "  Running Prettier in check mode"
    fi
    find . -name "*.$file_ext" -type f | grep -v "$exclude_pattern" | xargs pnpm exec prettier "$prettier_flag"
  else
    if [ "$mode" = "check" ]; then
      echo "  Checking all files (no exclusions)"
    else
      echo "  Formatting all files (no exclusions)"
    fi
    find . -name "*.$file_ext" -type f | xargs pnpm exec prettier "$prettier_flag"
  fi
}

# Parse common formatting script arguments
parse_format_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --check)
        FORMAT_MODE="check"
        shift
        ;;
      --write)
        FORMAT_MODE="write"
        shift
        ;;
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
        if [ -n "${EXCLUDE_HELP_TEXT:-}" ]; then
          echo "$EXCLUDE_HELP_TEXT"
        fi
        exit 1
        ;;
    esac
  done
}
