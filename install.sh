#!/bin/sh
set -u
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)
            if [ $# -lt 2 ]; then
                echo "Error: --profile requires an argument"
                exit 1
            fi
            PROFILE="--profile=$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="$1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script:"
            echo "  1. Installs uv and python dependencies"
            echo "  2. Configures Metta for your profile"
            echo "  3. Installs components"
            echo ""
            echo "Options:"
            echo "  --profile PROFILE      Set user profile (external, cloud, or softmax)"
            echo "                         If not specified, runs interactive configuration"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Interactive setup"
            echo "  $0 --profile=softmax   # Setup for Softmax employee"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
done


err() {
    echo "ERROR: $1" >&2
    exit 1
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Welcome to Metta!"

if ! check_cmd uv; then
    echo "\nuv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add all common uv installation paths to PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    # Source env files if they exist
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

    if ! check_cmd uv; then
        err "Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
    fi
fi

cd "$SCRIPT_DIR" || err "Failed to change to project directory"
uv sync || err "Failed to install Python dependencies"
uv run python -m metta.setup.metta_cli symlink-setup || err "Failed to set up metta command in ~/.local/bin"
uv run python -m metta.setup.metta_cli configure $PROFILE || err "Failed to run configuration"
uv run python -m metta.setup.metta_cli install || err "Failed to install components"

echo "\nSetup complete!\n"

if ! check_cmd metta; then
    echo "To start using metta, ensure ~/.local/bin is in your PATH:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.) to make it permanent."
fi
