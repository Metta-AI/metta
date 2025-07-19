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
This builds the C++ extension with debug symbols and no optimization.

### Step 2: Set Breakpoints
- Open any `.hpp` file (e.g., `src/metta/mettagrid/agent.hpp`)
- Click to the left of line numbers to set breakpoints
- Breakpoints appear as red dots

### Step 3: Start Debugging
- Open VS Code/Cursor
- Select `C++ Debug: Train Metta` from the debug dropdown (or `C++ Debug: Test` for tests)
- Press F5 or click the green play button

### Step 4: Debug
- Execution will stop at your breakpoints
- Hover over variables to see values
- Use debug controls:
  - F10: Step over
  - F11: Step into
  - Shift+F11: Step out
  - F5: Continue

## Example

1. Open `agent.hpp`
2. Find the `attack()` method
3. Set a breakpoint inside the method
4. Run `C++ Debug: Train Metta`
5. When an agent attacks, execution stops at your breakpoint

## Troubleshooting

### Breakpoints not working?
- Ensure you ran `./mettagrid/install-debug.sh` (not just `uv sync`)
- Check that the breakpoint is on an executable line (not a comment or declaration)
- Try rebuilding: `rm -rf _skbuild && ./mettagrid/install-debug.sh`

### Can't see variable values?
- Make sure you're in a debug build (`./mettagrid/install-debug.sh`)
- Try adding a temporary variable: `int debug_energy = this->energy;`

### Build errors?
- Clean and rebuild: `rm -rf _skbuild build dist && ./mettagrid/install-debug.sh`

## Tips

- **Debug builds are slower** - use regular `uv sync` for normal development
- **Printf debugging still works**: `std::cerr << "Debug: " << variable << std::endl;`
- **Check if debug build**: Look for `-O0` in the build output

## For the Team

- **Normal development**: Use `uv sync` (fast, optimized)
- **Debugging C++**: Use `./mettagrid/install-debug.sh` then VS Code's debug interface
- **Back to normal**: Just run `uv sync` again
