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

# More robust uv detection function
find_uv() {
    # First try command -v
    if command -v uv &> /dev/null; then
        return 0
    fi

    # Check common installation paths directly
    for path in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv"; do
        if [ -x "$path" ]; then
            # Found uv, export it directly
            export UV_BIN="$path"
            return 0
        fi
    done

    return 1
}

# Install uv if not present (required by setup_dev.sh)
if ! find_uv; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add all common uv installation paths to PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    # Source env files if they exist
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    # Force shell to rescan PATH (helps in some environments)
    hash -r 2>/dev/null || true

    # Verify installation
    if ! find_uv; then
        echo "❌ Failed to install uv"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv already available"
fi

# Use UV_BIN if set (from direct path detection), otherwise use 'uv'
UV="${UV_BIN:-uv}"

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
"$UV" run python -c "
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
echo "  • Run tests:         $UV run pytest"
echo "  • Format code:       $UV run ruff format && $UV run ruff check"
echo ""
echo "📚 Documentation: https://github.com/Metta-AI/metta"
echo "💬 Discord: https://discord.gg/mQzrgwqmwy"
echo ""
echo "🔧 To run commands, use: $UV run <command>"
echo "   Example: $UV run python -c 'import metta; print(metta.__file__)'"
echo ""
