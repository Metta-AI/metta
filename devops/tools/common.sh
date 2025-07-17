#!/bin/bash
# Common shell functions used across various scripts

err() {
  echo "ERROR: $1" >&2
  exit 1
}

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

# Get the absolute path of the script's directory
get_script_dir() {
  echo "$(cd "$(dirname "$0")" && pwd)"
}

# Get the project root directory (assumes this file is in devops/tools/)
get_project_root() {
  local script_dir="$(get_script_dir)"
  echo "$(cd "$script_dir/../.." && pwd)"
}

# Setup UV project environment
setup_uv_project_env() {
  local project_root="$(get_project_root)"
  export UV_PROJECT_ENVIRONMENT="$(cd "${project_root}/.venv" && pwd)"
}

# Setup UV environment paths
setup_uv_paths() {
  # Common locations where uv might be installed
  UV_PATHS="$HOME/.local/bin/uv $HOME/.cargo/bin/uv /opt/homebrew/bin/uv /usr/local/bin/uv"

  # Add directories containing uv to PATH (only if not already present)
  for uv_path in $UV_PATHS; do
    dir=$(dirname "$uv_path")
    case ":$PATH:" in
      *":$dir:"*) ;; # Already in PATH
      *) export PATH="$dir:$PATH" ;;
    esac
  done

  # Source env files if they exist
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

  # Force shell to rescan PATH (helps in some environments)
  hash -r 2> /dev/null || true
}

# Ensure uv is in PATH, installed, and uv project environment associated with this repo
ensure_uv_setup() {
  if ! check_cmd uv; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    setup_uv_paths

    if ! check_cmd uv; then
      err "Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
    fi
  fi
  setup_uv_project_env
}
