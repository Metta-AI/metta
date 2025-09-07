# Tribal Environment

A high-performance multi-agent environment implemented in Nim with Python bindings for reinforcement learning research.

## Overview

The tribal environment is a multi-agent gridworld featuring:
- Village-based agent tribes with shared altars (15 agents, compile-time constant)  
- Multi-step resource chains (ore → battery → hearts)
- Crafting system (wood → spears, wheat → hats/food, ore → armor)
- Defensive gameplay against Clippy enemies
- Terrain interaction (water, wheat fields, forests)

## Architecture

```
Nim Tribal Environment
         ↓
  genny bindings generator
         ↓
Generated Python bindings (tribal.py + libtribal.so)
         ↓
Python wrapper (tribal_genny.py)
         ↓
Training infrastructure
```

## Key Benefits of Genny Bindings

1. **High Performance**: Direct Nim→Python object mapping without serialization overhead
2. **Clean API**: Generates Pythonic interfaces with proper naming conventions  
3. **Automatic Memory Management**: Handles Nim GC integration seamlessly
4. **Type Safety**: Preserves type information across language boundaries

Compared to JSON-based IPC:
- **~10-100x faster** step times (no serialization overhead)  
- **Lower memory usage** (direct object sharing)
- **Better CPU cache behavior** (contiguous memory layout)
- **Reduced GC pressure** (fewer temporary objects)

## Build Process

```bash
# Install genny (one-time setup)
nimble install genny

# Generate Python bindings
./build_bindings.sh
```

This creates:
- `bindings/generated/tribal.py` - Python interface
- `bindings/generated/libtribal.so` - Compiled Nim library

## Usage

### Interactive Nim Application

```bash
# Run the interactive tribal environment
cd tribal/
nim r -d:release src/tribal
```

### Python Training Integration

```python
from metta.sim.tribal_genny import make_tribal_env

# Create environment
env = make_tribal_env(num_agents=15, max_steps=1000)

# Standard RL interface
obs, info = env.reset()
obs, rewards, terminals, truncations, info = env.step(actions)
```

### Direct Python Bindings

```python
import sys, os
sys.path.insert(0, 'bindings/generated')
import tribal

# Create environment with config
config = tribal.default_tribal_config()
env = tribal.TribalEnv(config)

# Use environment
env.reset_env()
obs = env.get_observations()
```

## API Overview

### TribalConfig
- `game.maxSteps`: Maximum steps per episode
- `game.enableCombat`: Enable/disable combat system
- `game.heartReward`, `oreReward`, `batteryReward`: Reward shaping
- `desyncEpisodes`: Desynchronize episode resets

### TribalEnv Methods
- `reset_env()`: Reset environment, return observations
- `step(actions)`: Step with actions, return success status
- `get_observations()`: Get current observations [agents][layers][height][width]
- `get_rewards()`: Get per-agent rewards for current step
- `get_terminated()`, `get_truncated()`: Episode status
- `get_current_step()`: Current step count
- `is_episode_done()`: Check if episode should end
- `render_text()`: Get text rendering of current state

### Action Space
- Action format: `[action_type, argument]` 
- Action types: 0=noop, 1=move, 2=attack, 3=get, 4=swap, 5=put
- Arguments: 0-7 for 8-directional actions (N,S,W,E,NW,NE,SW,SE)

### Observation Space
- Shape: `[agents, layers, height, width]` = `[15, 19, 11, 11]`
- Layers include: agents, inventory items, buildings, terrain features
- All values are uint8 (0-255)

## Testing

### Running Tests

#### Prerequisites
1. Build the bindings:
   ```bash
   ./build_bindings.sh
   ```

2. Activate the uv environment:
   ```bash
   source /home/relh/Code/metta/.venv/bin/activate  # if using uv
   ```

#### Test Commands
```bash
# Run the comprehensive test suite
python tests/test_python_bindings.py

# Run with unittest for more detailed output
python -m unittest tests.test_python_bindings -v

# Test training integration  
uv run ./tools/run.py experiments.recipes.tribal_basic.train run=test_genny

# Run Nim unit tests
nim r tests/test_unified_systems.nim
```

#### Test Coverage
The test suite (`tests/test_python_bindings.py`) covers:

- **Constants**: Environment dimensions, agent count, observation shape
- **Environment Management**: Creation, reset, stepping, termination
- **Actions**: All 6 action types (NOOP, MOVE, ATTACK, GET, SWAP, PUT)
- **Observations**: 15 agents × 19 layers × 11×11 grid = 34,485 elements
- **Rewards**: Multi-agent reward collection and validation
- **Sequences**: Genny sequence operations (SeqInt, SeqFloat, SeqBool)
- **Error Handling**: Global error system with check_error() and take_error()
- **Memory Management**: Object creation/destruction without leaks
- **Multi-Environment**: Concurrent environment instances
- **RL Integration**: Realistic training scenario simulation

#### Expected Results
All 14 tests should pass:
```
🎉 All tests passed! Tribal Python bindings are fully functional.
```

## Troubleshooting

### "Could not import Tribal bindings"
- Run `./build_bindings.sh` to generate bindings
- Check that `genny` is installed: `nimble list -i`
- Ensure the shared library (`libtribal.so`) exists in `bindings/generated/`
- Verify you're using the correct Python environment

### "Environment step failed"
- Check action format: actions should be `[num_agents, 2]` with int values
- Verify action types are in range 0-5
- Ensure arguments are in range 0-7

### Performance Issues
- Ensure using the genny bindings (`tribal_genny.py`) not manual C bindings
- Check that observations are being accessed efficiently (avoid frequent Python↔Nim conversions)
- Profile with `cProfile` to identify bottlenecks

### Performance Notes
- **Zero-copy observations**: Direct memory access between Nim and Python
- **Reward generation**: Environment produces ~0.001 rewards for GET actions
- **Step timing**: Each step processes 15 agents across 100×50 map
- **Memory efficient**: Automatic cleanup via genny's reference counting

## Development

### File Structure

**Core Environment:**
- `src/tribal/environment.nim` - Core game logic and mechanics
- `src/tribal/common.nim` - Shared types and constants  
- `src/tribal/terrain.nim` - Map generation and terrain features
- `src/tribal/objects.nim` - Game object definitions and placement
- `src/tribal/utils.nim` - Utility functions

**GUI Application:**
- `src/tribal.nim` - Main interactive application
- `src/tribal/renderer.nim` - Graphics rendering
- `src/tribal/ui.nim` - User interface components
- `src/tribal/panels.nim` - UI panel system
- `src/tribal/controls.nim` - Input handling

**AI & Combat:**
- `src/tribal/ai.nim` - AI controller system for automated agents
- `src/tribal/combat.nim` - Combat mechanics (if separate from environment)

**Other Systems:**
- `src/tribal/replays.nim` - Replay recording/playback
- `tests/` - Comprehensive test suite for all systems

**Python Integration:**
- `bindings/tribal_bindings.nim` - Genny bindings definition
- `bindings/generated/` - Generated Python bindings
- `build_bindings.sh` - Build script for bindings

## Future Improvements

1. **Runtime Configuration**: Make map size and agent count configurable at runtime
2. **Advanced Rendering**: Add visual rendering support through bindings  
3. **Parallel Environments**: Support multiple environment instances for vectorized training
4. **Custom Rewards**: Allow Python-defined reward functions
5. **Curriculum Support**: Dynamic environment parameter adjustment