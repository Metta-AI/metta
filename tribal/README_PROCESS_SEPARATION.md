# Tribal Environment Process Separation

This document explains the new process-separated architecture for the tribal environment that eliminates SIGSEGV crashes.

## Problem Solved

The previous nimpy-based approach caused SIGSEGV crashes when trying to initialize OpenGL contexts from within Python. The new architecture completely separates the Python neural network process from the Nim visualization process.

## Architecture Overview

### Components

1. **Nim Headless Server** (`tribal_headless_server`) - **RECOMMENDED**
   - Standalone Nim executable optimized for training
   - No graphics dependencies - eliminates all SIGSEGV issues
   - Runs the tribal environment simulation efficiently
   - Communicates via JSON files

2. **Nim Process Viewer** (`tribal_process_viewer`) - **EXPERIMENTAL**
   - Standalone Nim executable with OpenGL rendering
   - May experience SIGSEGV crashes due to graphics initialization
   - Use only when visualization is absolutely required

2. **Python Process Controller** (`tribal_process_controller.py`)
   - Controls neural network inference
   - Sends actions to the Nim process
   - Receives environment state from the Nim process
   - No OpenGL or graphics dependencies

3. **File-Based IPC**
   - `tribal_control.json`: Control messages (start/stop/ready)
   - `tribal_actions.json`: Actions from Python to Nim
   - `tribal_state.json`: Environment state from Nim to Python

### Communication Flow

```
Python Process          File System           Nim Process
    |                       |                      |
    |--- actions.json ----->|----> reads -------->|
    |                       |                      |
    |<--- state.json <------|<---- writes ---------|
    |                       |                      |  
    |--- control.json ----->|----> reads -------->|
```

## Usage

### Quick Test

Test the process communication directly:
```bash
cd tribal
python3 tribal_process_controller.py
```

### Using the Recipe

Test with the updated tribal recipe:
```bash
# Test with simple move actions
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move

# Test with noop actions  
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_noop

# Test with trained neural network
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=file://./checkpoints/policy.pt
```

## Benefits

1. **No SIGSEGV crashes**: Complete process separation eliminates OpenGL conflicts
2. **Native Nim performance**: Visualization runs at full native speed
3. **Easy debugging**: Each process can be debugged independently
4. **Robust communication**: JSON-based IPC with timeout handling
5. **Backward compatibility**: Old nimpy approach still available as `play_nimpy()`

## Technical Details

### File Formats

**Actions File (tribal_actions.json)**:
```json
{
  "actions": [[1, 2], [0, 0], [1, 1], ...],  // [action_type, arg] for each agent
  "timestamp": 1234567890.123
}
```

**State File (tribal_state.json)**:
```json
{
  "observations": [[[...]], ...],  // [agents][layers][height][width]
  "rewards": [0.1, -0.05, ...],    // Per-agent rewards
  "terminals": [false, false, ...], // Per-agent termination
  "truncations": [false, false, ...], // Per-agent truncation
  "current_step": 42,
  "max_steps": 1000,
  "episode_done": false,
  "timestamp": 1234567890.123
}
```

**Control File (tribal_control.json)**:
```json
{
  "active": true,      // Whether Python is actively controlling
  "shutdown": false,   // Signal to shutdown Nim process
  "ready": true,       // Nim process ready signal
  "timestamp": 1234567890.123
}
```

### Error Handling

- **Timeout Protection**: Python waits max 2 seconds for state updates
- **File Cleanup**: Communication files are removed on shutdown
- **Process Management**: Graceful termination with fallback to force-kill
- **Stale Data Prevention**: Files are deleted after reading

### Performance Optimization

- **JSON Streaming**: Files are written atomically
- **Polling Frequency**: 50ms polling for responsive communication  
- **File Sizes**: State files are ~500KB for typical observations
- **Frame Rate**: Nim viewer runs at 30 FPS by default

## Comparison with Previous Approaches

| Aspect | Nimpy (Old) | Process Separation (New) |
|--------|-------------|--------------------------|
| SIGSEGV Issues | ✗ Frequent crashes | ✅ Eliminated |
| OpenGL Conflicts | ✗ Context sharing issues | ✅ Isolated processes |
| Debugging | ✗ Mixed stack traces | ✅ Clear separation |
| Performance | ✅ Fast | ✅ Fast (similar) |
| Memory Usage | ✅ Lower | ⚠️ Slightly higher |
| Setup Complexity | ✅ Simple | ⚠️ Two processes |

## Migration Guide

### For Existing Code

Replace:
```python
# Old approach (may crash)
play_tool = play_nimpy(env_config, policy_uri)

# New approach (stable)
play_tool = play(env_config, policy_uri)
```

### For New Development

Use the new `play()` function by default:
```python
# Recommended - uses process separation
result = play(policy_uri="test_move")

# Legacy - only if you specifically need nimpy
result = play_nimpy(policy_uri="test_move") 
```

## Building

Build the headless server (recommended for training):
```bash
cd tribal
./build_headless_server.sh
```

Or build the process viewer (experimental - may crash with SIGSEGV):
```bash
cd tribal
./build_process_viewer.sh
```

## Troubleshooting

### Common Issues

1. **"Viewer executable not found"**
   - Run `./build_process_viewer.sh` in the tribal directory

2. **"Timeout waiting for state update"**
   - Check that the Nim process is running
   - Verify tribal assets are present in `data/` directory

3. **"Failed to start Nim process"**
   - Ensure all Nim dependencies are installed
   - Check that the current directory has write permissions

4. **"Communication files not found"**
   - Verify you're running from the tribal directory
   - Check file permissions

### Debug Mode

Enable debug output by modifying the controller:
```python
controller = TribalProcessController(tribal_dir, debug=True)
```

### Performance Tuning

Adjust polling frequency in `tribal_process_controller.py`:
```python
time.sleep(0.05)  # 50ms polling (20 Hz)
time.sleep(0.02)  # 20ms polling (50 Hz) - higher CPU usage
```

## Future Improvements

1. **Named Pipes**: Replace file-based IPC with named pipes for better performance
2. **Protobuf**: Use binary serialization instead of JSON for larger observations
3. **Shared Memory**: For ultra-high performance scenarios
4. **WebSocket**: For remote/distributed training setups