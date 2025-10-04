# GAMMA Alignment Metrics - Complete Implementation

## Overview

This module implements the complete GAMMA (General Alignment Metric for Multi-agent Autonomy) framework for Metta AI, providing quantitative measures of multi-agent alignment.

**Status**: ✅ **Production Ready** - All 5 core metrics implemented and tested with real MettaGrid

## What's Implemented

### Core Metrics (5/5) ✅

1. **Goal Attainment (A_i)** - `metrics/goal_attainment.py`
2. **Directional Intent (D_i)** - `metrics/directional_intent.py`
3. **Path Efficiency (E_i)** - `metrics/path_efficiency.py`
4. **Time Efficiency (T_i)** - `metrics/time_efficiency.py`
5. **Energy Proportionality (Y_i)** - `metrics/energy_proportionality.py`

### Collective Metrics ✅

- **Individual Alignment Metric (IAM)** - Geometric mean of all 5 components
- **GAMMA** - Huber mean aggregation across swarm
- **GAMMA_α** - Dispersion-penalized version

### Misalignment Detectors ✅

- **Anti-progress mass** - Detects motion against task direction
- **Loopiness** - Detects detours and loops

### Integration with Metta ✅

- **TrajectoryCollector** - Collects agent data during episodes
- **MettaGridAdapter** - Extracts positions and computes task directions from MettaGrid
- **GAMMAEvaluator** - Computes metrics from trajectories
- **GAMMALogger** - TrainerComponent for automatic logging

### Task Interfaces

- **SetpointTask** - Goal-reaching tasks (single or multiple goals)
- **TaskInterface** - Base class for custom tasks

## Testing

All components tested and verified:

- ✅ Unit-level: `simple_alignment_demo.py` - Tests all 5 metrics with simulated agents
- ✅ Integration: `test_with_real_mettagrid.py` - Tests with actual MettaGrid environment

## Usage

### Quick Start

```python
from metta.alignment.integration import GAMMALogger

# Add to trainer
gamma_logger = GAMMALogger(
    num_agents=16,
    epoch_interval=10,
    alpha=0.1,
    enabled=True
)
trainer.add_component(gamma_logger)
```

### Manual Evaluation

```python
from metta.alignment.integration import (
    TrajectoryCollector,
    MettaGridAdapter,
    GAMMAEvaluator
)

# Collect trajectories
collector = TrajectoryCollector(num_agents=4)
adapter = MettaGridAdapter()

for step in range(episode_length):
    positions = adapter.extract_agent_positions(env)
    task_dirs = adapter.compute_task_directions_to_resources(env)
    collector.record_step(positions, task_directions=task_dirs, dt=0.1)

# Evaluate
trajectories = collector.get_trajectories()
evaluator = GAMMAEvaluator(alpha=0.1)
results = evaluator.evaluate(trajectories, dt=0.1)

print(f"GAMMA: {results['GAMMA']:.3f}")
```

## File Structure

```
metta/alignment/
├── metrics/                    # Core metric implementations
│   ├── base.py                # Base class
│   ├── directional_intent.py # D_i metric
│   ├── path_efficiency.py    # E_i metric
│   ├── goal_attainment.py    # A_i metric
│   ├── time_efficiency.py    # T_i metric
│   ├── energy_proportionality.py  # Y_i metric
│   └── gamma.py              # IAM and GAMMA
├── task_interfaces/           # Task specifications
│   ├── base.py               # Base interface
│   └── setpoint.py           # Goal-reaching tasks
├── integration/              # Metta-specific integration
│   ├── trajectory_collector.py
│   ├── mettagrid_adapter.py
│   ├── gamma_evaluator.py
│   ├── gamma_logger.py
│   └── README.md
├── examples/                 # Working examples
│   ├── simple_alignment_demo.py
│   └── test_with_real_mettagrid.py
└── README.md                # Main documentation
```

## Key Features

- **Framework-agnostic**: Works with any RL framework
- **Observable-only**: No access to internal policy needed
- **Tested**: Verified with real MettaGrid environment
- **Production-ready**: Type-annotated, documented, linted
- **Extensible**: Easy to add new metrics or task interfaces

## References

- Paper: "GAMMA: A Framework-Agnostic Alignment Metric for Autonomous Swarms" by Marcel Blattner and Adam Goldstein
- TAS Framework: "Tangential Action Spaces" by Marcel Blattner
- Metta AI: https://github.com/Metta-AI/metta

## Next Steps

To use in production:
1. Add `GAMMALogger` to your trainer configuration
2. Define task-specific direction computation for your scenarios
3. Calibrate baseline parameters from honest runs
4. Monitor alignment metrics during training

## Testing

Run tests:
```bash
uv run python metta/alignment/examples/simple_alignment_demo.py
uv run python metta/alignment/examples/test_with_real_mettagrid.py
```
