#!/bin/sh
# shellcheck shell=dash
# shellcheck disable=SC2039  # local is non-POSIX
#
# Metta initial setup script
# This script ensures uv is installed and sets up the Python environment

set -u

# Variables for output control
PRINT_VERBOSE=${METTA_INSTALLER_PRINT_VERBOSE:-0}
PRINT_QUIET=${METTA_INSTALLER_PRINT_QUIET:-0}
NO_MODIFY_PATH=${METTA_NO_MODIFY_PATH:-0}

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
for arg in "$@"; do
    case "$arg" in
        --no-modify-path)
            NO_MODIFY_PATH=1
            ;;
        --verbose|-v)
            PRINT_VERBOSE=1
            ;;
        --quiet|-q)
            PRINT_QUIET=1
            ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

Options:
    -v, --verbose
            Enable verbose output

    -q, --quiet
            Disable progress output

        --no-modify-path
            Don't configure the PATH environment variable

    -h, --help
            Print help information
EOF
            exit 0
            ;;
        *)
            OPTIND=1
            if [ "${arg%%--*}" = "" ]; then
                err "unknown option $arg"
            fi
            while getopts :hvq sub_arg "$arg"; do
                case "$sub_arg" in
                    h)
                        "$0" --help
                        exit 0
                        ;;
                    v)
                        PRINT_VERBOSE=1
                        ;;
                    q)
                        PRINT_QUIET=1
                        ;;
                    *)
                        err "unknown option -$OPTARG"
                        ;;
                esac
            done
            ;;
    esac
done

INFERRED_HOME=$(get_home)
INFERRED_HOME_EXPRESSION=$(get_home_expression)

say "Welcome to Metta!"
say "This script will set up your development environment."
say ""

# Check if uv is installed
if ! check_cmd uv; then
    say "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    say "uv has been installed."
    say ""
fi

# Install Python dependencies
say "Installing Python dependencies..."
cd "$PROJECT_DIR"
ensure uv sync

# BIN_DIR is in metta/setup/installer/bin
BIN_DIR="$PROJECT_DIR/metta/setup/installer/bin"
BIN_DIR_EXPR="$(replace_home "$BIN_DIR")"

# Ensure the wrapper script is executable
ensure chmod +x "$BIN_DIR/metta"

# Check if metta is already accessible in PATH
check_metta_accessible() {
    if command -v metta > /dev/null 2>&1; then
        local existing_metta
        existing_metta="$(command -v metta)"

        # Make sure it's actually an executable file, not a directory
        if [ ! -f "$existing_metta" ] || [ ! -x "$existing_metta" ]; then
            return 1  # Not a valid executable
        fi

        # Check if it's our metta (could be the actual file or a symlink to it)
        if [ "$existing_metta" = "$BIN_DIR/metta" ]; then
            return 0  # Direct match
        fi

        # Check if it's the venv shim that runs the same thing
        if [ "$existing_metta" = "$PROJECT_DIR/.venv/bin/metta" ]; then
            # The venv metta is just a shim that runs our metta_cli, so it's effectively the same
            return 0
        fi

        # Check if they resolve to the same target (handles symlinks)
        if [ -e "$existing_metta" ] && [ -e "$BIN_DIR/metta" ]; then
            local existing_target=$(cd "$(dirname "$existing_metta")" && pwd -P)/$(basename "$existing_metta")
            local our_target=$(cd "$(dirname "$BIN_DIR/metta")" && pwd -P)/$(basename "$BIN_DIR/metta")
            if [ "$existing_target" = "$our_target" ]; then
                return 0
            fi
        fi
    fi
    return 1  # metta is not in PATH or points elsewhere
}

# Check if metta is already properly set up
if check_metta_accessible; then
    say ""
    say "metta is already installed and accessible in your PATH."
    say "Setup complete!"
    exit 0
fi

# Avoid modifying the users PATH if they are managing their PATH manually
case :$PATH:
  in *:$BIN_DIR:*) NO_MODIFY_PATH=1 ;;
     *) ;;
esac

say "Installing to $BIN_DIR"

# PATH configuration - using uv installer's approach
if [ "$NO_MODIFY_PATH" = "0" ]; then
    # Create env scripts
    ENV_SCRIPT="$BIN_DIR/env"
    FISH_ENV_SCRIPT="$BIN_DIR/env.fish"
    ENV_SCRIPT_EXPR="$(replace_home "$ENV_SCRIPT")"
    FISH_ENV_SCRIPT_EXPR="$(replace_home "$FISH_ENV_SCRIPT")"

    say ""
    say "Setting up PATH configuration..."

    # Add to CI PATH first
    add_to_ci_path "$BIN_DIR"

    # Use uv's approach: write to multiple shell configs
    add_install_dir_to_path "$BIN_DIR_EXPR" "$ENV_SCRIPT" "$ENV_SCRIPT_EXPR" ".profile" "sh"
    exit1=$?
    shotgun_install_dir_to_path "$BIN_DIR_EXPR" "$ENV_SCRIPT" "$ENV_SCRIPT_EXPR" ".profile .bashrc .bash_profile .bash_login" "sh"
    exit2=$?
    add_install_dir_to_path "$BIN_DIR_EXPR" "$ENV_SCRIPT" "$ENV_SCRIPT_EXPR" ".zshrc .zshenv" "sh"
    exit3=$?
    # This path may not exist by default
    ensure mkdir -p "$INFERRED_HOME/.config/fish/conf.d"
    exit4=$?
    add_install_dir_to_path "$BIN_DIR_EXPR" "$FISH_ENV_SCRIPT" "$FISH_ENV_SCRIPT_EXPR" ".config/fish/conf.d/metta.fish" "fish"
    exit5=$?

    if [ "${exit1:-0}" = 1 ] || [ "${exit2:-0}" = 1 ] || [ "${exit3:-0}" = 1 ] || [ "${exit4:-0}" = 1 ] || [ "${exit5:-0}" = 1 ]; then
        say ""
        say "To add $BIN_DIR_EXPR to your PATH, either restart your shell or run:"
        say ""
        say "    source $ENV_SCRIPT_EXPR (sh, bash, zsh)"
        say "    source $FISH_ENV_SCRIPT_EXPR (fish)"
    fi
else
    say ""
    say "Skipping PATH configuration. To use metta directly, add this to your shell profile:"
    say "  export PATH=\"$BIN_DIR:\$PATH\""
fi

# Check for shadowed binaries
_shadowed_bins="$(check_for_shadowed_bins "$BIN_DIR" "metta")"
if [ -n "$_shadowed_bins" ]; then
    warn "The following commands are shadowed by other commands in your PATH:$_shadowed_bins"
fi

say ""
say "Setup complete!"

# Show usage examples
if [ "$NO_MODIFY_PATH" = "1" ] && ! check_metta_accessible; then
    say ""
    say "To use the 'metta' command, you have several options:"
    say ""
    say "Option 1: Add to PATH manually:"
    say "  export PATH=\"$BIN_DIR_EXPR:\$PATH\""
    say "  metta configure"
    say "  metta install"
    say "  metta status"
    say ""
    say "Option 2: Activate the virtual environment first:"
    say "  source .venv/bin/activate"
    say "  metta configure"
    say "  metta install"
    say "  metta status"
fi
