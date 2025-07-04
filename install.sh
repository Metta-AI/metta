#!/bin/sh
# Metta initial setup script
# This script ensures uv is installed and runs the metta setup

set -u

# Parse command line arguments
NO_MODIFY_PATH=""
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-modify-path)
            NO_MODIFY_PATH="--no-modify-path"
            shift
            ;;
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
            echo "This script sets up the Metta development environment by:"
            echo "  1. Installing uv (if not already installed)"
            echo "  2. Installing Python dependencies"
            echo "  3. Setting up PATH configuration (optional)"
            echo "  4. Configuring Metta for your profile"
            echo "  5. Installing components for your profile"
            echo ""
            echo "Options:"
            echo "  --profile PROFILE      Set user profile (external, cloud, or softmax)"
            echo "                         If not specified, runs interactive configuration"
            echo "  --no-modify-path       Don't modify shell configuration files"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Interactive setup"
            echo "  $0 --profile=softmax   # Setup for Softmax employee"
            echo "  $0 --no-modify-path    # Setup without modifying PATH"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Simple output functions
say() {
    echo "$1"
}

err() {
    echo "ERROR: $1" >&2
    exit 1
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

say "Welcome to Metta!"
say "This script will set up your development environment."
say ""

# Check if uv is installed
if ! check_cmd uv; then
    say "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source uv env to make it available in current session
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi

    # Verify uv is now available
    if ! check_cmd uv; then
        err "Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
    fi

    say "uv has been installed successfully."
    say ""
fi

# Install Python dependencies
say "Installing Python dependencies..."
cd "$SCRIPT_DIR" || err "Failed to change to project directory"
uv sync || err "Failed to install Python dependencies"

# Run metta path setup to ensure metta is in PATH
say ""
say "Setting up metta command..."
uv run python -m metta.setup.metta_cli path-setup $NO_MODIFY_PATH || err "Failed to set up PATH"

# Run metta configure
say ""
if [ -n "$PROFILE" ]; then
    say "Configuring Metta with specified profile..."
else
    say "Running metta configuration wizard..."
fi
uv run python -m metta.setup.metta_cli configure $PROFILE || err "Failed to run configuration"

# Run metta install
say ""
say "Installing configured components..."
uv run python -m metta.setup.metta_cli install || err "Failed to install components"

say ""
say "Setup complete!"
say ""

if [ -n "$NO_MODIFY_PATH" ]; then
    say "To use metta, add the following to your shell profile:"
    say "  export PATH=\"$SCRIPT_DIR/metta/setup/installer/bin:\$PATH\""
    say ""
    say "Then you can use commands like:"
else
    say "To start using metta, either:"
    say "  1. Restart your terminal, or"
    say "  2. Run: source ~/.metta/env"
    say ""
    say "Then you can use commands like:"
fi

say "  metta status    # Check component status"
say "  metta install   # Install additional components"
