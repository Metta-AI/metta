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
    echo "✓ LLDB found (macOS debugger)"
    echo "Note: Make sure to use 'lldb' instead of 'gdb' in your launch.json"
else
    # Linux
    if ! command -v gdb &> /dev/null; then
        echo "Error: GDB not found. Please install it:"
        echo "  Ubuntu/Debian: sudo apt-get install gdb"
        echo "  Fedora: sudo dnf install gdb"
        echo "  Arch: sudo pacman -S gdb"
        exit 1
    fi
    echo "✓ GDB found (Linux debugger)"
fi

# Change to mettagrid directory
cd "$(dirname "$0")"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build build-debug build-debug-gdb _skbuild dist

# Use the debug-gdb preset for maximum debugging capability
echo "Configuring with debug-gdb preset..."
export SKBUILD_CMAKE_ARGS="--preset debug-gdb"

# Build with debug configuration
echo "Building with debug symbols..."
uv pip install -e . -v --config-settings=cmake.build-type="Debug" \
                      --config-settings=install.strip=false

# Create symlink for compile_commands.json (helps with IDE support)
echo "Setting up IDE support..."
if [ -d "_skbuild" ]; then
    COMPILE_COMMANDS=$(find _skbuild -name "compile_commands.json" | head -1)
    if [ -n "$COMPILE_COMMANDS" ]; then
        if [[ "$COMPILE_COMMANDS" == "$PWD/_skbuild/"* ]]; then
            ln -sf "$COMPILE_COMMANDS" compile_commands.json
            echo "✓ Created compile_commands.json symlink"
        else
            echo "⚠️ Found compile_commands.json at unexpected location: $COMPILE_COMMANDS"
            echo "⚠️ Symlink not created for security reasons"
        fi
        echo "✓ Created compile_commands.json symlink"
    fi
fi

echo ""
echo "🎉 Debug build complete!"
echo ""
echo "Next steps:"
echo "1. Open VS Code/Cursor"
echo "2. Set breakpoints in .hpp/.cpp files"
echo "3. Use 'C++ Debug: Train Metta' from the debug dropdown"
echo "4. Press F5 to start debugging"
echo ""
echo "Tip: Breakpoints work best on executable lines (not declarations or comments)"
