# C++ Debugging Guide for MettaGrid

## One-Time Setup

1. Make the debug script executable:
```bash
chmod +x mettagrid/install-debug.sh
```

2. Install debugger:
```bash
# Ubuntu/Debian
sudo apt-get install gdb

# macOS
# LLDB comes with Xcode Command Line Tools:
xcode-select --install
# Note: You'll need to modify launch.json (see below)
```

### macOS Configuration
If on macOS, modify the launch configurations in `.vscode/launch.json`:
- Change `"MIMode": "gdb"` to `"MIMode": "lldb"`
- Remove the `"miDebuggerPath"` line
- Remove GDB-specific `setupCommands`

## How to Debug C++ Code

### Step 1: Build with Debug Symbols
```bash
./mettagrid/install-debug.sh
```
This builds the C++ extension using the `debug-gdb` preset with:
- **Maximum debug info** (`-g3 -ggdb`)
- **No optimization** (`-O0`)
- **Full stack traces** (`-fno-omit-frame-pointer`)
- **No function inlining** (`-fno-inline`)

### Step 2: Set Breakpoints
- Open any `.hpp` or `.cpp` file (e.g., `src/metta/mettagrid/agent.hpp`)
- Click to the left of line numbers to set breakpoints
- Breakpoints appear as red dots
- **Tip**: Set breakpoints on executable lines, not declarations or comments

### Step 3: Start Debugging
- Open VS Code/Cursor
- Select `C++ Debug: Train Metta` from the debug dropdown (or `C++ Debug: Test` for tests)
- Press F5 or click the green play button

### Step 4: Debug
- Execution will stop at your breakpoints
- Hover over variables to see values
- Use debug controls:
  - **F10**: Step over (next line)
  - **F11**: Step into (enter function)
  - **Shift+F11**: Step out (exit function)
  - **F5**: Continue execution
  - **F9**: Toggle breakpoint

### Advanced Debugging Features

#### Variable Inspection
- **Watch window**: Add variables to watch their values change
- **Call stack**: See the complete function call hierarchy
- **Memory view**: Inspect raw memory (useful for pointers/arrays)

#### Conditional Breakpoints
- Right-click on a breakpoint → "Edit Breakpoint"
- Add conditions like `energy < 50` or `id == 42`

## Example Debugging Session

1. Open `src/metta/mettagrid/agent.hpp`
2. Find the `attack()` method
3. Set a breakpoint inside the method (on a line that actually executes)
4. Run `C++ Debug: Train Metta`
5. When an agent attacks, execution stops at your breakpoint
6. Inspect variables like `this->energy`, `target`, etc.
7. Step through the code to see how the attack logic works

## Build Configurations

We have multiple CMake presets available:

- **`debug`**: Basic debug build
- **`debug-gdb`**: Enhanced debug build (used by install-debug.sh)
- **`coverage`**: Debug build with code coverage
- **`release`**: Optimized release build
- **`release-no-tests`**: Release build without tests

## Troubleshooting

### Breakpoints not working?
- ✅ Ensure you ran `./mettagrid/install-debug.sh` (not just `uv sync`)
- ✅ Check that the breakpoint is on an executable line (not a comment or declaration)
- ✅ Try rebuilding: `rm -rf _skbuild && ./mettagrid/install-debug.sh`
- ✅ Verify debug symbols: Look for `-g3 -O0` in the build output

### Can't see variable values?
- ✅ Make sure you're in a debug build (`./mettagrid/install-debug.sh`)
- ✅ Try adding a temporary variable: `int debug_energy = this->energy;`
- ✅ Check if variables are optimized away (shouldn't happen with `-O0`)

### Build errors?
- ✅ Clean and rebuild: `rm -rf _skbuild build dist && ./mettagrid/install-debug.sh`
- ✅ Check CMake preset: `cmake --list-presets`
- ✅ Verify compiler flags in build output

### VS Code/Cursor Issues?
- ✅ Ensure `compile_commands.json` exists in project root
- ✅ Reload window: Cmd/Ctrl+Shift+P → "Developer: Reload Window"
- ✅ Check C++ extension is installed and enabled

## Performance Notes

- **Debug builds are 2-5x slower** - use regular `uv sync` for normal development
- **File size**: Debug binaries are much larger due to symbol information
- **Memory usage**: Debug builds use more memory

## Alternative Debugging Methods

### Printf Debugging
Still works great for quick debugging:
```cpp
std::cerr << "Debug: energy=" << this->energy << ", position=(" << x << "," << y << ")" << std::endl;
```

### GDB Command Line
You can also debug directly with GDB:
```bash
gdb python
(gdb) run -c "import mettagrid; mettagrid.train()"
```

### Core Dumps
Enable core dumps for post-mortem debugging:
```bash
ulimit -c unlimited
# Run your program, if it crashes:
gdb python core
```

## For the Team

- **Normal development**: Use `uv sync` (fast, optimized)
- **Debugging C++**: Use `./mettagrid/install-debug.sh` then VS Code's debug interface
- **Back to normal**: Just run `uv sync` again
- **CI/Testing**: Use `cmake --preset release` for performance tests

## Quick Reference

| Task | Command |
|------|---------|
| Normal development | `uv sync` |
| Debug build | `./mettagrid/install-debug.sh` |
| Clean rebuild | `rm -rf _skbuild && ./mettagrid/install-debug.sh` |
| List CMake presets | `cmake --list-presets` |
| Check compile commands | `ls -la compile_commands.json` |
