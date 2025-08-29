# Sweep Orchestrator Guide

## Overview

The new sweep orchestrator provides a stateless, distributed-friendly hyperparameter optimization system for Metta. It uses Protein (Bayesian optimization) to efficiently explore hyperparameter spaces and WandB for persistent state management.

## Quick Start

### Running a Quick Test Sweep

```bash
# Run a quick test sweep with 5 trials
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    --args sweep_name=my_test_sweep max_trials=5
```

### Running a Full PPO Hyperparameter Sweep

```bash
# Run a full PPO sweep with 10 trials
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    --args sweep_name=ppo_sweep_001 max_trials=10
```

## Architecture

The sweep system consists of four main components:

1. **Orchestrator** (`metta/sweep/sweep_orchestrator.py`): Stateless controller that coordinates the sweep
2. **Scheduler** (`metta/sweep/scheduler/`): Decides which jobs to run next
   - `OptimizingScheduler`: Uses Protein optimizer for Bayesian optimization
   - `SequentialScheduler`: Simple one-job-at-a-time scheduler
3. **Store** (`metta/sweep/store/wandb.py`): Persistent state management via WandB
4. **Dispatcher** (`LocalDispatcher`): Executes jobs as local subprocesses

## Creating Custom Sweeps

### Example: Custom PPO Sweep

```python
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep_orchestrator import SweepOrchestratorTool

def my_custom_sweep(
    sweep_name: str = None,
    max_trials: int = 20,
) -> SweepOrchestratorTool:
    """Create a custom hyperparameter sweep."""
    
    # Define parameters to sweep
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",  # Metric to optimize
        goal="maximize",
        method="bayes",  # Use Bayesian optimization
        parameters={
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=1e-3,
                scale="auto",
            ),
            "trainer.ppo.clip_coef": ParameterConfig(
                min=0.05,
                max=0.3,
                distribution="uniform",
                mean=0.175,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=5,  # Random samples before Bayesian optimization
            max_suggestion_cost=300,  # Max seconds per trial
        ),
    )
    
    return SweepOrchestratorTool(
        sweep_name=sweep_name,
        protein_config=protein_config,
        max_trials=max_trials,
        recipe_module="experiments.recipes.arena",
        train_entrypoint="train_shaped",
        eval_entrypoint="evaluate",
        max_parallel_jobs=1,  # Sequential execution
    )
```

## Parameter Configuration

### Distribution Types

- **`log_normal`**: For parameters that vary over orders of magnitude (e.g., learning rates)
- **`uniform`**: For parameters with linear relationships (e.g., clip coefficients)

### Required Fields

Each `ParameterConfig` requires:
- `min`: Minimum value
- `max`: Maximum value  
- `distribution`: Distribution type
- `mean`: Mean value (geometric mean for log_normal)
- `scale`: Scale parameter (use "auto" for automatic scaling)

## Monitoring Sweeps

### Via Logs

The orchestrator provides detailed logging:
```
[SweepOrchestrator] Starting sweep: my_sweep
[SweepOrchestrator] Dispatching trial_0001: trainer.optimizer.learning_rate=0.001
[SweepOrchestrator] Trial trial_0001 completed with score: 0.85
```

### Via WandB

All sweep runs are grouped in WandB by sweep name. View them at:
```
https://wandb.ai/YOUR_ENTITY/YOUR_PROJECT/groups/SWEEP_NAME
```

## Advanced Configuration

### Parallel Execution

```python
SweepOrchestratorTool(
    max_parallel_jobs=4,  # Run 4 jobs in parallel
    monitoring_interval=10,  # Check job status every 10 seconds
)
```

### Custom Training Overrides

```python
SweepOrchestratorTool(
    train_overrides={
        "trainer.total_timesteps": "100000",  # Limit training steps
        "trainer.num_workers": "2",  # Number of parallel workers
    }
)
```

### Optimization Methods

The Protein optimizer supports three methods:
- `"bayes"`: Bayesian optimization (default, most efficient)
- `"random"`: Random search
- `"genetic"`: Genetic algorithm for multi-objective optimization

## Resuming Sweeps

The orchestrator is stateless and can be safely interrupted. To resume:

```bash
# Simply run the same command again
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    --args sweep_name=ppo_sweep_001 max_trials=10
```

The orchestrator will:
1. Fetch existing runs from WandB
2. Continue from where it left off
3. Schedule remaining trials

## Troubleshooting

### Common Issues

1. **WandB authentication**: Ensure `WANDB_API_KEY` is set
2. **Subprocess failures**: Check individual run logs in `train_dir/SWEEP_NAME/`
3. **Protein suggestions**: If optimizer fails, it falls back to random search

### Debug Mode

For debugging, modify `LocalDispatcher` in `sweep_orchestrator.py`:
```python
# Comment out these lines to see subprocess output
# stdout=subprocess.DEVNULL,
# stderr=subprocess.DEVNULL,
```

## Migration from Old Sweep System

The new orchestrator is a complete replacement for the old Hydra-based sweep system:

### Key Differences

| Old System | New System |
|------------|------------|
| Hydra configuration | Pydantic configuration |
| Stateful controller | Stateless orchestrator |
| Single machine only | Distributed-ready |
| Basic grid/random search | Bayesian optimization with Protein |
| File-based state | WandB-based persistent state |

### Benefits

- **Stateless**: Can be interrupted and resumed safely
- **Efficient**: Bayesian optimization converges faster than random search
- **Observable**: Full visibility via WandB and structured logging
- **Extensible**: Easy to add new schedulers and optimizers
- **Testable**: Clean separation of concerns with protocols

## Examples in the Codebase

- `experiments/sweeps/standard.py`: Standard sweep configurations
  - `ppo()`: Full PPO hyperparameter sweep
  - `quick_test()`: Quick test configuration

## API Reference

See the docstrings in:
- `metta/tools/sweep_orchestrator.py`: Main tool interface
- `metta/sweep/sweep_orchestrator.py`: Core orchestration logic
- `metta/sweep/protein_config.py`: Parameter configuration
- `metta/sweep/scheduler/optimizing.py`: Optimizer-based scheduling