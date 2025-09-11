#!/bin/bash
# Build Tribal Environment from Metta Root Directory

set -e  # Exit on any error

echo "🔧 Building Tribal Environment from Metta Root..."

# Get absolute paths
METTA_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRIBAL_DIR="$METTA_ROOT/tribal"

echo "Metta root: $METTA_ROOT"
echo "Tribal directory: $TRIBAL_DIR"

# Check that tribal directory exists
if [ ! -d "$TRIBAL_DIR" ]; then
    echo "❌ Error: Tribal directory not found at $TRIBAL_DIR"
    exit 1
fi

# Check that tribal build script exists
TRIBAL_BUILD_SCRIPT="$TRIBAL_DIR/build_bindings.sh"
if [ ! -f "$TRIBAL_BUILD_SCRIPT" ]; then
    echo "❌ Error: Tribal build script not found at $TRIBAL_BUILD_SCRIPT"
    exit 1
fi

echo "🔨 Running tribal build script..."
cd "$TRIBAL_DIR"
./build_bindings.sh

echo "✅ Tribal environment built successfully from root!"
echo ""
echo "You can now use tribal from Python:"
echo "  from tribal.src.tribal_genny import make_tribal_env"
echo "  env = make_tribal_env()"