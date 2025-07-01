#!/bin/sh
# Metta initial setup script
# This script ensures uv is installed and sets up the Python environment

set -e

# Get the actual location of this script (resolving symlinks)
SCRIPT_PATH="$0"
while [ -h "$SCRIPT_PATH" ]; do
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
    SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
    # Handle relative symlinks
    case "$SCRIPT_PATH" in
        /*) ;;
        *) SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH" ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# Get the project root (three levels up from metta/setup/installer/)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source the install utilities from the same directory as the actual script
. "$SCRIPT_DIR/install_utils.sh"

# Parse command line arguments
ADD_TO_PATH_ARG=""
for arg in "$@"; do
    case "$arg" in
        --add-to-path)
            ADD_TO_PATH_ARG=1
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --add-to-path    Automatically add metta to PATH without prompting"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
    esac
done

echo "Welcome to Metta!"
echo "This script will set up your development environment."
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    echo "uv has been installed."
    echo
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# BIN_DIR is in metta/setup/installer/bin
BIN_DIR="$PROJECT_DIR/metta/setup/installer/bin"

# Ensure the wrapper script is executable
chmod +x "$BIN_DIR/metta"



# Check if metta is already accessible in PATH
check_metta_accessible() {
    if command -v metta > /dev/null 2>&1; then
        local existing_metta
        existing_metta="$(command -v metta)"
        if [ "$existing_metta" = "$BIN_DIR/metta" ]; then
            return 0  # metta is already in PATH and points to our installation
        fi
    fi
    return 1  # metta is not in PATH or points elsewhere
}

# PATH configuration - adapted from uv installer patterns
if [ -n "$ADD_TO_PATH_ARG" ]; then
    # Command line flag was provided
    ADD_TO_PATH=$ADD_TO_PATH_ARG
    if [ "$ADD_TO_PATH" = "1" ]; then
        echo
        echo "Adding metta to PATH as requested..."
    fi
elif check_metta_accessible; then
    # metta is already in PATH
    ADD_TO_PATH=0
    echo
    echo "metta is already accessible in your PATH."
else
    # Ask the user
    echo
    echo "Would you like to add the metta command to your PATH?"
    echo "This will allow you to use 'metta' directly from any directory."
    printf "Add to PATH? [Y/n] "
    read REPLY
    echo

    case "$REPLY" in
        [nN]*)
            ADD_TO_PATH=0
            ;;
        *)
            ADD_TO_PATH=1
            ;;
    esac
fi

if [ "$ADD_TO_PATH" = "1" ]; then
    # Create env scripts
    ENV_SCRIPT="$BIN_DIR/env"
    FISH_ENV_SCRIPT="$BIN_DIR/env.fish"

    echo
    echo "Setting up PATH configuration..."

    # Create the env scripts
    write_env_script_sh "$BIN_DIR" "$ENV_SCRIPT"
    write_env_script_fish "$BIN_DIR" "$FISH_ENV_SCRIPT"

    # Detect shell and apply modifications
    detected_shell=$(detect_shell)
    echo "Detected shell: ${detected_shell:-sh}"
    echo

    # Apply PATH modifications to appropriate config files
    if apply_path_modifications "$BIN_DIR" "$ENV_SCRIPT" "$FISH_ENV_SCRIPT" "$detected_shell"; then
        ENV_SCRIPT_EXPR=$(replace_home_with_var "$ENV_SCRIPT")
        FISH_ENV_SCRIPT_EXPR=$(replace_home_with_var "$FISH_ENV_SCRIPT")

        echo
        echo "PATH configuration complete!"
        echo
        echo "To use 'metta' in this terminal session:"
        if [ "$detected_shell" = "fish" ]; then
            echo "  source \"$FISH_ENV_SCRIPT_EXPR\""
        else
            echo "  source \"$ENV_SCRIPT_EXPR\""
        fi
        echo
        echo "Or open a new terminal window."
    else
        echo
        echo "PATH already configured."
    fi
else
    echo
    echo "Skipping PATH configuration. To use metta directly, add this to your shell profile:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

# Check for shadowed binaries
check_shadowed_binary "metta" "$BIN_DIR/metta"

# Add to CI PATH if applicable
add_to_ci_path "$BIN_DIR"

echo
echo "Setup complete!"

if [ "$ADD_TO_PATH" = "1" ]; then
    # User added to PATH - already shown instructions above
    :
else
    echo
    echo "To use the 'metta' command, you have several options:"
    echo
    echo "Option 1: Add to PATH manually:"
    echo "  export PATH=\"$PROJECT_DIR/metta/setup/installer/bin:\$PATH\""
    echo "  metta configure"
    echo "  metta install"
    echo "  metta status"
    echo
    echo "Option 2: Activate the virtual environment first:"
    echo "  source .venv/bin/activate"
    echo "  metta configure"
    echo "  metta install"
    echo "  metta status"
fi
