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


_find_project_root() {
  # When sourced from install.sh, SCRIPT_DIR is already the project root
  if [ -n "${SCRIPT_DIR:-}" ]; then
    echo "$SCRIPT_DIR"
  else
    # Fallback for other scripts - use the directory of the calling script
    echo "$(cd "$(dirname "$0")/../.." && pwd)"
  fi
}

PROJECT_ROOT="$(_find_project_root)"

# Setup UV project environment
setup_uv_project_env() {
  export UV_PROJECT_ENVIRONMENT="$(cd "${PROJECT_ROOT}/.venv" && pwd)"
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
  # hash -r is bash-specific, use command -v to verify uv is findable instead
  command -v uv > /dev/null 2>&1 || true
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
