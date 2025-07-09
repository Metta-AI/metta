#!/bin/bash
# OpenHands setup script for Metta AI
# This script automatically sets up the development environment when starting an OpenHands conversation

set -e
set -o pipefail

echo "🚀 Setting up Metta AI development environment for OpenHands..."

# Get the project directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📁 Working in: $PROJECT_DIR"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: Not in Metta project root directory"
    exit 1
fi

# Set up git configuration for OpenHands first
echo "🔧 Configuring git..."
git config --global user.name "openhands" 2>/dev/null || true
git config --global user.email "openhands@all-hands.dev" 2>/dev/null || true

# Common locations where uv might be installed
UV_PATHS="$HOME/.local/bin/uv $HOME/.cargo/bin/uv /opt/homebrew/bin/uv /usr/local/bin/uv"

# Install uv if not present (required by setup_dev.sh)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add directories containing uv to PATH (only if not already present)
    for uv_path in $UV_PATHS; do
        dir=$(dirname "$uv_path")
        case ":$PATH:" in
            *":$dir:"*) ;;  # Already in PATH
            *) export PATH="$dir:$PATH" ;;
        esac
    done

    # Source env files if they exist
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    # Force shell to rescan PATH (helps in some environments)
    hash -r 2>/dev/null || true

    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv already available"
fi

# Set the project environment to ensure we use the correct venv
export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"

# Use the existing comprehensive setup script
echo "🛠️  Running Metta development setup script..."
echo "ℹ️  This will install all dependencies and configure the environment"

# Set environment variable to indicate we're in Docker/container (OpenHands environment)
export IS_DOCKER=1

# Run the official setup script
if bash devops/setup_dev.sh; then
    echo "✅ Development setup completed successfully"
else
    echo "⚠️  Setup script completed with warnings (this may be normal)"
fi

# Quick verification
echo "🧪 Verifying installation..."
uv run python -c "
try:
    import metta
    print('✅ Core metta package imported successfully')
    try:
        import metta.mettagrid
        print('✅ Metta mettagrid module imported successfully')
    except ImportError as e:
        print(f'⚠️  Mettagrid module import issue: {e}')

    try:
        import metta.rl.fast_gae
        print('✅ C++ extensions (fast_gae) imported successfully')
    except ImportError as e:
        print(f'⚠️  C++ extensions import issue: {e}')

    print('✅ Setup verification completed - Metta is ready to use!')
except ImportError as e:
    print(f'❌ Critical error - Metta package not found: {e}')
    print('Setup may have failed. Please check the output above for errors.')
    exit 1
" 2>/dev/null

# Display helpful information
echo ""
echo "🎉 Metta AI setup complete!"
echo ""
echo "📋 Quick start commands:"
echo "  • Train a model:     ./tools/train.py run=my_experiment +hardware=macbook wandb=off"
echo "  • Play interactively: ./tools/play.py run=my_experiment +hardware=macbook wandb=off"
echo "  • Run tests:         uv run pytest"
echo "  • Format code:       uv run ruff format && uv run ruff check"
echo ""
echo "📚 Documentation: https://github.com/Metta-AI/metta"
echo "💬 Discord: https://discord.gg/mQzrgwqmwy"
echo ""
echo "🔧 To run commands, use: uv run <command>"
echo "   Example: uv run python -c 'import metta; print(metta.__file__)'"
echo ""
