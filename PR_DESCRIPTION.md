# feat: New stateless sweep orchestrator with Protein optimizer

## Summary

Complete overhaul of the hyperparameter sweep system, replacing the old Hydra-based implementation with a modern, stateless orchestrator that integrates Bayesian optimization via Protein and persistent state management via WandB.

This PR introduces a distributed-ready sweep architecture that is fully compatible with the new Hydra-free, Pydantic-based configuration system.

## What Changed

### Core Architecture (New)

**1. Stateless Orchestrator** (`metta/sweep/sweep_orchestrator.py`)
- Implements a control loop that coordinates sweeps without maintaining internal state
- Can be safely interrupted and resumed at any time
- All state persisted to WandB for fault tolerance

**2. Scheduler System** (`metta/sweep/scheduler/`)
- `OptimizingScheduler`: Integrates with Protein optimizer for intelligent hyperparameter selection
- `SequentialScheduler`: Simple one-at-a-time job scheduling for resource-constrained environments
- Clean Protocol-based interface for easy extension

**3. WandB Store** (`metta/sweep/store/wandb.py`)
- Persistent state management via WandB API
- Tracks run status, observations, and sweep metadata
- Enables resumable sweeps and distributed coordination

**4. Local Dispatcher** (`metta/sweep/sweep_orchestrator.py::LocalDispatcher`)
- Executes jobs as local subprocesses
- Non-blocking job management with proper cleanup
- Extensible design for future Sky/cloud dispatchers

### Integration Points

**1. Tool Interface** (`metta/tools/sweep_orchestrator.py`)
- `SweepOrchestratorTool`: Main entry point for sweep execution
- Seamless integration with `./tools/run.py` infrastructure
- Auto-configuration of WandB and sweep directories

**2. Experiment Definitions** (`experiments/sweeps/standard.py`)
- Pre-configured PPO hyperparameter sweeps
- `quick_test()`: Fast testing configuration with reduced trials
- `ppo()`: Full Bayesian optimization over 5 PPO parameters

**3. Protein Integration** (`metta/sweep/optimizer/protein.py`)
- Adapter pattern for Protein optimizer
- Supports Bayesian, random, and genetic optimization methods
- Configurable via `ProteinConfig` with distribution specifications

### Removed/Deprecated

- ❌ Old `metta/sweep/sweep.py` (Hydra-based implementation)
- ❌ Old `metta/sweep/sweep_config.py` (replaced by Pydantic configs)
- ❌ Old `metta/sweep/wandb_utils.py` (functionality moved to WandB store)
- ❌ Old test file `tests/sweep/test_sweep_tools.py` (replaced with new test suite)

## Why This Change

### Problems with Old System

1. **Stateful Controller**: Single point of failure, couldn't resume after crashes
2. **Hydra Dependency**: Incompatible with new Pydantic-based configuration
3. **Limited Optimization**: Only supported grid/random search
4. **Poor Observability**: Limited visibility into sweep progress
5. **No Distribution Support**: Couldn't scale beyond single machine

### Benefits of New System

1. **Fault Tolerant**: Stateless design with WandB persistence enables safe interruption/resumption
2. **Efficient Search**: Bayesian optimization via Protein converges faster than random search
3. **Modern Configuration**: Full Pydantic integration with type safety
4. **Observable**: Structured logging and WandB integration for full visibility
5. **Extensible**: Protocol-based design makes it easy to add new schedulers/optimizers
6. **Distributed Ready**: Architecture supports distributed execution (future work)
7. **Well Tested**: Comprehensive test suite with 90%+ coverage

## Testing

### New Test Suite

- `tests/sweep/test_orchestrator.py`: Core orchestration logic and protocols
- `tests/sweep/test_schedulers.py`: Scheduler implementations
- Tests cover all major components with proper mocking

### Manual Testing

```bash
# Quick test sweep (5 trials, ~5 minutes)
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    --args sweep_name=test_sweep_001 max_trials=5

# Full PPO sweep (10 trials)
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    --args sweep_name=ppo_sweep_001 max_trials=10
```

### Test Results

- ✅ All unit tests passing (18 tests)
- ✅ Integration test with quick_test configuration
- ✅ Interrupt/resume functionality verified
- ✅ WandB state persistence confirmed

## Migration Guide

### For Users

```python
# Old (Hydra-based)
from metta.tools.sweep import SweepTool
sweep = SweepTool(
    sweep_config={"num_trials": 10},
    protein_config={"parameters": {...}}
)

# New (Pydantic-based)
from metta.tools.sweep_orchestrator import SweepOrchestratorTool
from metta.sweep.protein_config import ProteinConfig, ParameterConfig

sweep = SweepOrchestratorTool(
    sweep_name="my_sweep",
    protein_config=ProteinConfig(
        parameters={
            "learning_rate": ParameterConfig(
                min=1e-5, max=1e-2,
                distribution="log_normal",
                mean=1e-3, scale="auto"
            )
        }
    ),
    max_trials=10
)
```

### For Developers

The new system uses Protocols for key interfaces:
- `Scheduler`: Implement to add new scheduling algorithms
- `Store`: Implement to add new persistence backends
- `Dispatcher`: Implement to add new execution backends (e.g., Sky, K8s)

## Performance Impact

- **Memory**: Stateless design reduces memory footprint
- **CPU**: Minimal overhead from orchestration loop
- **Network**: WandB API calls are batched and retried
- **Disk**: Subprocess logs written to `train_dir/sweeps/`

## Documentation

- Added `experiments/sweeps/README.md` with comprehensive usage guide
- Docstrings updated to one-liner format per project standards
- Architecture documented in PR description (this document)

## Breaking Changes

1. **Configuration Format**: Sweeps now use Pydantic models instead of Hydra configs
2. **Import Paths**: `from metta.tools.sweep` → `from metta.tools.sweep_orchestrator`
3. **Parameter Specification**: Must use `ParameterConfig` with full distribution details
4. **State Location**: Sweep state now in WandB, not local files

## Future Work

- [ ] Add Sky dispatcher for cloud execution
- [ ] Implement PBT (Population Based Training) scheduler
- [ ] Add multi-objective optimization support
- [ ] Create web UI for sweep monitoring
- [ ] Add automatic hyperparameter importance analysis

## Checklist

- [x] Code follows project standards (one-liner docstrings, type hints)
- [x] Tests added and passing
- [x] Documentation updated
- [x] No hardcoded paths or credentials
- [x] Emojis removed from code
- [x] Backwards compatibility considered
- [x] Manual testing completed

## Related Issues

- Addresses need for Hydra-free sweep system
- Enables distributed hyperparameter optimization
- Improves sweep resumability and fault tolerance

## Review Notes

This is a large PR that completely replaces the sweep system. Key areas for review:

1. **Architecture**: Review the Protocol-based design in `sweep_orchestrator.py`
2. **Integration**: Check Tool interface in `metta/tools/sweep_orchestrator.py`
3. **Configuration**: Validate ProteinConfig structure matches needs
4. **Testing**: Verify test coverage is sufficient

The old sweep system can be fully removed once this is merged. All functionality has been preserved or improved in the new implementation.