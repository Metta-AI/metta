#!/bin/bash
# Debug build script for MettaGrid
# Location: mettagrid/install-debug.sh

set -e

echo "Building MettaGrid with debug symbols..."

# Check for debugger availability
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v lldb &> /dev/null; then
        echo "Error: LLDB not found. Please install Xcode Command Line Tools:"
        echo "  xcode-select --install"
        exit 1
    fi
    echo "Note: On macOS, you'll need to use LLDB instead of GDB."
    echo "Make sure to change 'MIMode' from 'gdb' to 'lldb' in your launch.json"
else
    # Linux
    if ! command -v gdb &> /dev/null; then
        echo "Error: GDB not found. Please install it:"
        echo "  Ubuntu/Debian: sudo apt-get install gdb"
        echo "  Fedora: sudo dnf install gdb"
        echo "  Arch: sudo pacman -S gdb"
        exit 1
    fi
fi

# Change to mettagrid directory
cd "$(dirname "$0")"

# Clean previous builds
rm -rf build build-debug _skbuild dist

# Set debug environment
export SKBUILD_CMAKE_ARGS="--preset debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG='-g3 -ggdb -O0 -fno-omit-frame-pointer -fno-inline'"

# Build with debug configuration
pip install -e . -v --config-settings=cmake.build-type="Debug" \
                    --config-settings=install.strip=false

# Create symlink for compile_commands.json
if [ -d "_skbuild" ]; then
    COMPILE_COMMANDS=$(find _skbuild -name "compile_commands.json" | head -1)
    if [ -n "$COMPILE_COMMANDS" ]; then
        ln -sf "$COMPILE_COMMANDS" compile_commands.json
    fi
fi

echo "Debug build complete!"
echo "To debug: Use 'C++ Debug: Train Metta' (or similar) in VS Code"
