# MettaGrid Environment Library Integration Demos

This directory contains demonstration scripts showing how to use MettaGrid with different reinforcement learning libraries.

## Available Demos

### 1. PufferLib Integration (`puffer_demo.py`)
Demonstrates the `MettaGridPufferEnv` class which provides:
- Vectorized environment interface
- Efficient buffer management
- PufferLib-compatible data types
- Multi-agent support with shared buffers

**Usage:**
```bash
cd demos
python puffer_demo.py
```

### 2. Gymnasium Integration (`gym_demo.py`)
Demonstrates both `MettaGridGymEnv` and `SingleAgentMettaGridGymEnv` classes:
- Standard Gymnasium environment interface
- Both multi-agent and single-agent modes
- Compatible with Gym's `step()` and `reset()` methods
- Proper observation and action space definitions

**Usage:**
```bash
cd demos
python gym_demo.py
```

### 3. PettingZoo Integration (`pettingzoo_demo.py`)
Demonstrates the `MettaGridPettingZooEnv` class which provides:
- PettingZoo ParallelEnv interface
- Agent-centric observations and actions
- Dynamic agent management (agents can be removed when done)
- Support for heterogeneous multi-agent scenarios

**Usage:**
```bash
cd demos
python pettingzoo_demo.py
```

## Class Hierarchy

The implementation follows this hierarchy:

1. **MettaGridCore** - Stateless C++ environment wrapper
2. **MettaGridEnv** - Base Python environment with curriculum/stats support
3. **Framework-specific adapters**:
   - `MettaGridPufferEnv` - PufferLib adapter
   - `MettaGridGymEnv` - Gymnasium adapter
   - `MettaGridPettingZooEnv` - PettingZoo adapter

## Key Features

- **Unified Interface**: All adapters share the same underlying core functionality
- **Efficient Buffer Management**: Each adapter handles buffers optimally for its framework
- **Curriculum Support**: All environments support curriculum learning
- **Stats & Replay**: Integrated stats collection and replay recording
- **Multi-Agent Support**: Native multi-agent support across all frameworks

## Configuration

All demos use a simple configuration with:
- Basic navigation setup
- Random map generation
- Simple action space (noop, move, rotate)
- Configurable number of agents

For production use, you would typically use more complex configurations with:
- Terrain-based maps
- Resource collection mechanics
- Advanced action spaces
- Curriculum learning schedules

## Next Steps

For actual training, you would:

1. **For PufferLib**: Use with vectorized training loops
2. **For Gymnasium**: Integrate with stable-baselines3 or similar
3. **For PettingZoo**: Use with multi-agent training libraries like Ray RLlib

Each adapter is designed to be a drop-in replacement for standard environments in their respective frameworks.