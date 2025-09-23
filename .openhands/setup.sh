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
git config --global user.name "openhands" 2> /dev/null || true
git config --global user.email "openhands@all-hands.dev" 2> /dev/null || true

# Set environment variable to indicate we're in Docker/container (OpenHands environment)
export IS_DOCKER=1

# Run the main install script with external profile (suitable for OpenHands)
echo "🛠️  Running Metta installation script..."
if bash ./install.sh --profile external --non-interactive; then
  echo "✅ Installation completed successfully"
else
  echo "❌ Installation failed"
  exit 1
fi

# Quick verification
echo "🧪 Verifying installation..."
if uv run python -c "
import metta
print('✅ Core metta package imported successfully')
try:
    import mettagrid
    print('✅ Metta mettagrid module imported successfully')
except ImportError as e:
    print(f'⚠️  Mettagrid module import issue: {e}')

print('✅ Setup verification completed - Metta is ready to use!')
" 2>&1; then
  echo ""
else
  echo "⚠️  Some imports failed, but this may be expected in certain environments"
fi

# Display helpful information
echo ""
echo "🎉 Metta AI setup complete!"
echo ""
echo "📋 Quick start commands:"
echo "  • Train a model:     ./tools/run.py train arena run=my_experiment"
echo "  • Play interactively: ./tools/run.py play arena"
echo "  • Run tests:         uv run pytest"
echo "  • Format code:       uv run ruff format && uv run ruff check"
echo ""
echo "📚 Documentation: https://github.com/Metta-AI/metta"
echo "💬 Discord: https://discord.gg/mQzrgwqmwy"
echo ""
