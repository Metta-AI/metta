# Tribal Environment Python Bindings

This document describes how the Nim tribal environment is bound to Python using the `genny` library for high-performance RL training.

## Overview

The tribal environment is implemented in Nim for performance, but needs Python bindings for integration with the Metta training infrastructure. We use [genny](https://github.com/treeform/genny) to generate clean, high-performance Python bindings automatically.

## Architecture

```
Nim Tribal Environment
         ↓
  genny bindings generator
         ↓
Generated Python bindings (Tribal.py + Tribal.so)
         ↓
Python wrapper (tribal_genny.py)
         ↓
Training infrastructure
```

## Key Benefits of Genny

1. **High Performance**: Direct Nim→Python object mapping without serialization overhead
2. **Clean API**: Generates Pythonic interfaces with proper naming conventions
3. **Automatic Memory Management**: Handles Nim GC integration seamlessly
4. **Type Safety**: Preserves type information across language boundaries

## Build Process

```bash
# Install genny (one-time setup)
cd mettascope2
nimble install genny

# Generate Python bindings
./build_bindings.sh
```

This creates:
- `bindings/generated/Tribal.py` - Python interface
- `bindings/generated/Tribal.so` - Compiled Nim library

## Usage

```python
from metta.sim.tribal_genny import make_tribal_env

# Create environment
env = make_tribal_env(num_agents=15, max_steps=1000)

# Standard RL interface
obs, info = env.reset()
obs, rewards, terminals, truncations, info = env.step(actions)
```

## API Overview

### TribalConfig
- `numAgents`: Number of agents (currently fixed at compile time)  
- `maxSteps`: Maximum steps per episode
- `mapWidth`, `mapHeight`: Map dimensions
- `seed`: Random seed (for reproducibility)

### TribalEnv Methods
- `reset(seed)`: Reset environment, return observations
- `step(actions)`: Step with actions, return (obs, rewards, terminals, truncations, info)
- `getObservations()`: Get current observations [agents][layers][height][width]
- `getRewards()`: Get per-agent rewards for current step
- `getCurrentStep()`, `getMaxSteps()`: Episode progress
- `renderText()`: Get text rendering of current state

### Action Space
- Action format: `[action_type, argument]` 
- Action types: 0=noop, 1=move, 2=attack, 3=get, 4=swap, 5=put
- Arguments: 0-7 for 8-directional actions (N,S,W,E,NW,NE,SW,SE)

### Observation Space
- Shape: `[agents, layers, height, width]` = `[15, 19, 11, 11]`
- Layers include: agents, inventory items, buildings, terrain features
- All values are uint8 (0-255)

## Performance Characteristics

Compared to JSON-based IPC:
- **~10-100x faster** step times (no serialization overhead)  
- **Lower memory usage** (direct object sharing)
- **Better CPU cache behavior** (contiguous memory layout)
- **Reduced GC pressure** (fewer temporary objects)

## Testing

```bash
# Test basic functionality
python test_tribal_genny.py

# Test training integration  
uv run ./tools/run.py experiments.recipes.tribal_basic.train run=test_genny
```

## Troubleshooting

### "Could not import Tribal bindings"
- Run `./build_bindings.sh` to generate bindings
- Check that `genny` is installed: `nimble list -i`

### "Environment step failed"
- Check action format: actions should be `[num_agents, 2]` with int values
- Verify action types are in range 0-5
- Ensure arguments are in range 0-7

### Performance Issues
- Ensure using the genny bindings (`tribal_genny.py`) not manual C bindings
- Check that observations are being accessed efficiently (avoid frequent Python↔Nim conversions)
- Profile with `cProfile` to identify bottlenecks

## Future Improvements

1. **Runtime Configuration**: Make map size and agent count configurable at runtime
2. **Advanced Rendering**: Add visual rendering support through bindings  
3. **Parallel Environments**: Support multiple environment instances for vectorized training
4. **Custom Rewards**: Allow Python-defined reward functions
5. **Curriculum Support**: Dynamic environment parameter adjustment