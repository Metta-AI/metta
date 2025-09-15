# Tribal Environment

A high-performance multi-agent environment implemented in Nim with Python bindings for reinforcement learning research.

## Overview

The tribal environment is a multi-agent gridworld featuring:
- **15 agents** distributed across **3 villages** with shared altars 
- Multi-step resource chains (ore → battery → hearts)
- Crafting system (wood → spears, wheat → hats/food, ore → armor)
- **High-frequency Clippy adversaries** with mixed behavior (spawn every ~20 steps!)
- Terrain interaction (water, wheat fields, forests)

## Quick Start

### 1. Build the Environment
```bash
cd tribal/
./build_bindings.sh
```

### 2. Test Everything Works
```bash
python test_tribal_bindings.py
```

This self-contained test will verify clippy spawning, agent behavior, and environment functionality.

### 3. Train Agents
```bash
# Basic training
uv run ./tools/run.py experiments.recipes.tribal_basic.train

# Test with simple policies
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move
```

## Architecture

### Core Environment Architecture

```
Nim Tribal Environment
         ↓
  genny bindings generator
         ↓
Generated Python bindings (tribal.py + libtribal.dylib)
         ↓
Python wrapper (tribal_genny.py) - Zero-copy pointer interface
         ↓
Training infrastructure
```

**★ Insight ─────────────────────────────────────**
- The environment uses **direct pointer access** for zero-copy performance between Python and Nim
- Python handles memory allocation, Nim reads/writes directly to numpy arrays
- This eliminates data conversion overhead and significantly improves performance
**─────────────────────────────────────────────────**

### Process Separation for Interactive Use

For visualization and interactive testing, the environment supports process separation to avoid SIGSEGV crashes:

```
Python Process          File System           Nim Process
    |                       |                      |
    |--- actions.json ----->|----> reads -------->|
    |                       |                      |
    |<--- state.json <------|<---- writes ---------|
    |                       |                      |  
    |--- control.json ----->|----> reads -------->|
```

## Key Benefits

### Performance Advantages
1. **Zero-Copy Operations**: Direct pointer access between Python and Nim
2. **High Performance**: ~10-100x faster than JSON-based IPC
3. **Low Memory Usage**: Direct object sharing without duplication
4. **Better CPU Cache**: Contiguous memory layout
5. **Reduced GC Pressure**: Fewer temporary objects

### Stability Features  
1. **No SIGSEGV crashes**: Process separation eliminates OpenGL conflicts
2. **Native Nim performance**: Visualization runs at full native speed
3. **Easy debugging**: Each process can be debugged independently
4. **Robust communication**: JSON-based IPC with timeout handling

## Usage

### Python Training Integration

```python
from tribal.src.tribal_genny import make_tribal_env

# Create environment with zero-copy interface
env = make_tribal_env(num_agents=15, max_steps=1000)

# Standard RL interface
obs, info = env.reset()
obs, rewards, terminals, truncations, info = env.step(actions)
```

### Recipe-Based Training

```bash
# Basic training
uv run ./tools/run.py experiments.recipes.tribal_basic.train run=my_experiment

# Evaluation
uv run ./tools/run.py experiments.recipes.tribal_basic.evaluate policy_uri=file://./checkpoints/policy.pt

# Interactive testing (process-separated)
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move
```

### Direct Python Bindings

```python
import sys
sys.path.insert(0, 'tribal/bindings/generated')
import tribal

# Create environment with config
config = tribal.default_tribal_config()
env = tribal.TribalEnv(config)

# Use zero-copy interface
env.reset_env()
obs = env.get_observations()
```

## API Reference

### Environment Configuration

**TribalGameConfig**:
- `max_steps`: Maximum steps per episode (default: 2000)
- `enable_combat`: Enable/disable combat system (default: True)
- `ore_per_battery`: Ore required to craft battery (default: 3)
- `batteries_per_heart`: Batteries required at altar for hearts (default: 2)
- `clippy_spawn_rate`: Rate of enemy spawning (default: 1.0)
- `clippy_damage`: Damage dealt by enemies (default: 1)
- `heart_reward`, `ore_reward`, `battery_reward`: Reward shaping
- `survival_penalty`: Per-step survival penalty (default: -0.01)
- `death_penalty`: Penalty for agent death (default: -5.0)

**TribalEnvConfig**:
- `game`: TribalGameConfig instance
- `desync_episodes`: Desynchronize episode resets (default: True)
- `render_mode`: Rendering mode (default: None)

### Core Environment Methods

**TribalGridEnv** (simplified zero-copy interface):
- `reset(seed=None)`: Reset using direct pointer access
- `step(actions)`: Step using direct pointer access  
- `render(mode="human")`: Get text rendering
- `get_episode_stats()`: Get episode statistics

**TribalEnv** (direct Nim bindings):
- `reset_env()`: Reset environment
- `step(actions)`: Step with SeqInt actions
- `get_observations()`: Get current observations [agents][layers][height][width]
- `get_rewards()`: Get per-agent rewards
- `get_terminated()`, `get_truncated()`: Episode status
- `get_current_step()`: Current step count
- `is_episode_done()`: Check if episode should end
- `render_text()`: Get text rendering

### Action and Observation Spaces

**Action Space**:
- Format: `[action_type, argument]` 
- Action types: 0=noop, 1=move, 2=attack, 3=get, 4=swap, 5=put
- Arguments: 0-7 for 8-directional actions (N,S,W,E,NW,NE,SW,SE)

**Observation Space**:
- **Token format**: `[agents, max_tokens_per_agent, 3]` = `[15, 200, 3]` (zero-copy interface)
- **Grid format**: `[agents, layers, height, width]` = `[15, 19, 11, 11]` (direct bindings)
- Layers include: agents, inventory items, buildings, terrain features
- All values are uint8 (0-255)

## Testing

### Running Tests

#### Prerequisites
1. Build the bindings:
   ```bash
   ./build_bindings.sh
   ```

#### Test Commands
```bash
# Run the comprehensive test suite
python tests/test_python_bindings.py

# Run with unittest for more detailed output
python -m unittest tests.test_python_bindings -v

# Test training integration  
uv run ./tools/run.py experiments.recipes.tribal_basic.train run=test_genny

# Test process separation
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move

# Run Nim unit tests
nim r tests/test_unified_systems.nim
```

#### Test Coverage
The test suite covers:
- **Constants**: Environment dimensions, agent count, observation shape
- **Environment Management**: Creation, reset, stepping, termination
- **Actions**: All 6 action types (NOOP, MOVE, ATTACK, GET, SWAP, PUT)
- **Observations**: Token and grid formats
- **Rewards**: Multi-agent reward collection
- **Process Separation**: File-based IPC communication
- **Zero-Copy Interface**: Direct pointer access validation

## Interactive Play and Visualization

### Process-Separated Play

For interactive testing and visualization, use the process-separated approach to avoid crashes:

```bash
# Test with simple move actions (recommended)
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move

# Test with noop actions  
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_noop

# Test with trained neural network
uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=file://./checkpoints/policy.pt
```

### Building Interactive Components

Build the headless server (recommended for training):
```bash
cd tribal
./build_headless_server.sh
```

Or build the process viewer (experimental - may have SIGSEGV issues):
```bash
cd tribal
./build_process_viewer.sh
```

### Communication Files

The process separation uses JSON files for IPC:

- `tribal_actions.json`: Actions from Python to Nim
- `tribal_state.json`: Environment state from Nim to Python  
- `tribal_control.json`: Control messages (start/stop/ready)

## Build Process

### Required Dependencies

```bash
# Install Nim and nimble
curl https://nim-lang.org/choosenim/init.sh -sSf | sh

# Install genny (one-time setup)
nimble install genny
```

### Build Commands

```bash
# Generate Python bindings
./build_bindings.sh

# Build interactive components (optional)
./build_process_viewer.sh  # For visualization
./build_headless_server.sh # For training only
```

This creates:
- `bindings/generated/tribal.py` - Python interface
- `bindings/generated/libtribal.dylib` - Compiled Nim library
- `tribal_process_viewer` - Interactive viewer (optional)
- `tribal_headless_server` - Headless server (optional)

## Troubleshooting

### Common Build Issues

**"Could not import tribal bindings"**:
- Run `./build_bindings.sh` to generate bindings
- Check that `genny` is installed: `nimble list -i`
- Ensure the shared library exists in `bindings/generated/`
- Verify you're using the correct Python environment

**"Environment step failed"**:
- Check action format: actions should be `[num_agents, 2]` with int values
- Verify action types are in range 0-5
- Ensure arguments are in range 0-7

### Process Separation Issues

**"Timeout waiting for state update"**:
- Check that the Nim process is running
- Verify tribal assets are present in `data/` directory

**"Failed to start Nim process"**:
- Ensure all Nim dependencies are installed
- Check that the current directory has write permissions
- Build the process viewer: `./build_process_viewer.sh`

### Performance Issues

**Training performance**:
- Ensure using the zero-copy interface (`tribal_genny.py`)
- Check that observations are accessed efficiently
- Profile with `cProfile` to identify bottlenecks

**Interactive performance**:
- Use headless server for training (no graphics overhead)
- Adjust polling frequency in process controller if needed

## File Structure

### Core Environment
- `src/tribal/environment.nim` - Core game logic and mechanics
- `src/tribal/common.nim` - Shared types and constants  
- `src/tribal/terrain.nim` - Map generation and terrain features
- `src/tribal/objects.nim` - Game object definitions and placement
- `src/tribal/ai.nim` - AI controller system
- `src/tribal/external_actions.nim` - External neural network integration

### Python Integration
- `bindings/tribal_bindings.nim` - Genny bindings definition
- `bindings/generated/` - Generated Python bindings
- `src/tribal_genny.py` - Simplified zero-copy Python wrapper
- `src/tribal_process_controller.py` - Process separation controller
- `build_bindings.sh` - Build script for bindings

### Interactive Components
- `src/tribal.nim` - Main interactive application
- `src/tribal/renderer.nim` - Graphics rendering
- `src/tribal/ui.nim` - User interface components
- `build_process_viewer.sh` - Build script for viewer
- `build_headless_server.sh` - Build script for server

### Testing
- `tests/` - Comprehensive test suite
- `test_tribal_bindings.py` - Python bindings test
- `tests/test_unified_systems.nim` - Nim unit tests

## Performance Characteristics

### Zero-Copy Interface
- **Step timing**: ~640+ steps per second on M1 Mac
- **Memory efficient**: Pre-allocated numpy arrays, direct pointer access
- **Zero data copying**: Python and Nim share the same memory
- **Reward generation**: Environment produces rewards for resource collection

### Process Separation
- **File sizes**: State files are ~500KB for typical observations  
- **Frame rate**: Nim viewer runs at 30 FPS by default
- **Polling frequency**: 50ms polling for responsive communication
- **JSON streaming**: Files are written atomically

## Future Improvements

### Performance Optimizations
1. **Named Pipes**: Replace file-based IPC with named pipes
2. **Protobuf**: Use binary serialization for larger observations
3. **Shared Memory**: For ultra-high performance scenarios
4. **WebSocket**: For remote/distributed training setups

### Environment Features  
1. **Runtime Configuration**: Make map size and agent count configurable
2. **Advanced Rendering**: Add visual rendering support through bindings
3. **Parallel Environments**: Support multiple environment instances
4. **Custom Rewards**: Allow Python-defined reward functions
5. **Curriculum Support**: Dynamic environment parameter adjustment