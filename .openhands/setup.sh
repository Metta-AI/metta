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

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv already available"
fi

# Check Python version requirement
echo "🐍 Checking Python version..."
REQUIRED_PYTHON="3.11.7"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Found Python version: $PYTHON_VERSION"

# Install dependencies with uv
echo "📦 Installing Python dependencies..."
uv sync

# Verify critical imports (quick check)
echo "🔍 Verifying installation..."
if uv run python -c "
import sys
critical_imports = ['metta', 'numpy', 'torch', 'fastapi', 'wandb']
failed = []
for imp in critical_imports:
    try:
        __import__(imp)
        print(f'✅ {imp}')
    except ImportError:
        print(f'⚠️  {imp} (may be expected if not yet built)')
        failed.append(imp)
if len(failed) > 2:
    print(f'⚠️  Multiple imports failed: {failed}')
    sys.exit(1)
" 2>/dev/null; then
    echo "✅ Core dependencies verified"
else
    echo "⚠️  Some dependencies may need attention"
fi

# Try to build C++ extensions
echo "🔨 Building C++ extensions..."
if uv run python -c "import metta.rl.fast_gae" 2>/dev/null; then
    echo "✅ C++ extensions already built"
else
    echo "🔨 Building C++ extensions (this may take a moment)..."
    # The C++ extensions should be built automatically by uv sync due to scikit-build-core
    # If they're not, we can try to trigger a rebuild
    uv sync --reinstall-package metta 2>/dev/null || echo "⚠️  C++ build may need manual attention"
fi

# Set up git configuration for OpenHands
echo "🔧 Configuring git..."
git config --global user.name "openhands" 2>/dev/null || true
git config --global user.email "openhands@all-hands.dev" 2>/dev/null || true

# Create useful aliases and environment setup
echo "🛠️  Setting up development environment..."

# Check if we can import the main modules
echo "🧪 Final verification..."
if uv run python -c "
try:
    import metta
    import metta.mettagrid
    print('✅ Core Metta modules imported successfully')
except ImportError as e:
    print(f'⚠️  Some modules may need building: {e}')
"; then
    echo "✅ Setup verification passed"
fi

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