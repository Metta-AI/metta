# Sweep Orchestrator Guide

## Overview

The new sweep orchestrator provides a stateless, distributed-friendly hyperparameter optimization system for Metta. It uses Protein (Bayesian optimization) to efficiently explore hyperparameter spaces and WandB for persistent state management.

## Quick Start

### Running a Quick Test Sweep (Local)

```bash
# Run a quick test sweep with 5 trials locally
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    --args sweep_name=my_test_sweep max_trials=5 \
    --overrides dispatcher_type=local
```

### Running with Different Recipes

```bash
# Use arena_basic_easy_shaped recipe
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    --args \
    sweep_name=test_arena_basic \
    recipe_module=experiments.recipes.arena_basic_easy_shaped \
    train_entrypoint=train_shaped \
    eval_entrypoint=evaluate \
    max_trials=5
```

### Running a Full PPO Hyperparameter Sweep (Hybrid Mode)

```bash
# Run a full PPO sweep with 10 trials (default: train on cloud, eval locally)
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    --args sweep_name=ppo_sweep_001 max_trials=10
```

## Dispatcher Types

The sweep system supports three dispatcher modes:

- **`local`**: All jobs run locally with output capture
- **`skypilot`**: All jobs run on cloud resources via Skypilot
- **`hybrid_remote_train`** (default): Training on cloud, evaluation locally

```bash
# Force local execution
--overrides dispatcher_type=local

# Force cloud execution
--overrides dispatcher_type=skypilot

# Use hybrid mode (default)
--overrides dispatcher_type=hybrid_remote_train
```

## Configuration Structure Changes (Post-Merge)

After the recent merge, the PPO configuration structure has changed:

### Old Structure
```
trainer.ppo.clip_coef
trainer.ppo.ent_coef
trainer.ppo.gae_lambda
trainer.ppo.vf_coef
```

### New Structure
```
trainer.losses.loss_configs.ppo.clip_coef
trainer.losses.loss_configs.ppo.ent_coef
trainer.losses.loss_configs.ppo.gae_lambda
trainer.losses.loss_configs.ppo.vf_coef
```

The learning rate remains at: `trainer.optimizer.learning_rate`

## Architecture

The sweep system consists of four main components:

1. **Orchestrator** (`metta/sweep/controller.py`): Stateless controller that coordinates the sweep
2. **Scheduler** (`metta/sweep/scheduler/`): Decides which jobs to run next
   - `OptimizingScheduler`: Uses Protein optimizer for Bayesian optimization
   - `SequentialScheduler`: Simple one-job-at-a-time scheduler
3. **Store** (`metta/sweep/store/wandb.py`): Persistent state management via WandB
4. **Dispatcher**: Executes jobs
   - `LocalDispatcher`: Local subprocess execution with output capture
   - `SkypilotDispatcher`: Cloud execution via Skypilot
   - `RoutingDispatcher`: Routes different job types to different dispatchers

## Command-Line Arguments

The sweep tool now accepts these arguments via `--args`:

- `sweep_name`: Name of the sweep (auto-generated if not provided)
- `max_trials`: Maximum number of trials to run
- `recipe_module`: Recipe module to use (e.g., `experiments.recipes.arena`)
- `train_entrypoint`: Training function name (e.g., `train_shaped`)
- `eval_entrypoint`: Evaluation function name (e.g., `evaluate`)

Example with all arguments:
```bash
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    --args \
    sweep_name=my_custom_sweep \
    max_trials=10 \
    recipe_module=experiments.recipes.navigation \
    train_entrypoint=train \
    eval_entrypoint=evaluate
```

## Creating Custom Sweeps

### Example: Custom PPO Sweep with New Configuration

```python
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep import SweepOrchestratorTool

def my_custom_sweep(
    sweep_name: str = None,
    max_trials: int = 20,
) -> SweepOrchestratorTool:
    """Create a custom hyperparameter sweep."""
    
    # Define parameters to sweep (using new configuration structure)
    protein_config = ProteinConfig(
        metric="evaluator/eval_arena/score",  # Metric to optimize
        goal="maximize",
        method="bayes",  # Use Bayesian optimization
        parameters={
            # Learning rate remains in optimizer
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=1e-3,
                scale="auto",
            ),
            # PPO parameters now under losses.loss_configs.ppo
            "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
                min=0.05,
                max=0.3,
                distribution="uniform",
                mean=0.175,
                scale="auto",
            ),
            "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
                min=0.0001,
                max=0.01,
                distribution="log_normal",
                mean=0.001,
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
        dispatcher_type="hybrid_remote_train",  # Default dispatcher
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

### Real-time Monitoring Latency

The sweep monitor may show outdated progress compared to live training output. This is because:
- **WandB API latency**: The monitor polls WandB's API which has network and caching delays
- **Subprocess output**: Training logs stream directly in real-time

The subprocess output (e.g., "57.6% complete") is more current than the monitor table (e.g., "27.6% complete").

### Via Logs

The orchestrator provides detailed logging:
```
[SweepOrchestrator] Starting sweep: my_sweep
[OptimizingScheduler] 🚀 Scheduling trial 1/10: trial_0001
[OptimizingScheduler]    trainer.optimizer.learning_rate: 0.001
[OptimizingScheduler]    trainer.losses.loss_configs.ppo.clip_coef: 0.15
[trial_0001] Epoch 40 [axel.test_sweep.2038_trial_0001] / 798 sps / 46.08% of 50.00 ksteps
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
        "trainer.batch_size": "65536",  # Custom batch size
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
4. **Configuration errors**: Check that PPO parameters use the new path structure
5. **Progress monitoring lag**: Subprocess output is more current than WandB-based monitoring

### Debug Mode

To see subprocess output for debugging:

```bash
# Use local dispatcher with output capture
--overrides dispatcher_type=local capture_output=true
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
| `trainer.ppo.*` paths | `trainer.losses.loss_configs.ppo.*` paths |

### Benefits

- **Stateless**: Can be interrupted and resumed safely
- **Efficient**: Bayesian optimization converges faster than random search
- **Observable**: Full visibility via WandB and structured logging
- **Extensible**: Easy to add new schedulers and optimizers
- **Testable**: Clean separation of concerns with protocols
- **Flexible**: Support for different dispatchers and recipes

## Examples in the Codebase

- `experiments/sweeps/standard.py`: Standard sweep configurations
  - `ppo()`: Full PPO hyperparameter sweep
  - `quick_test()`: Quick test configuration with fewer trials

## Recent Changes

### Configuration Path Updates
- PPO parameters moved from `trainer.ppo.*` to `trainer.losses.loss_configs.ppo.*`
- This reflects the new loss system architecture where PPO is a configurable loss function

### New Command-Line Arguments
- Added support for `recipe_module`, `train_entrypoint`, `eval_entrypoint` as command-line args
- Default dispatcher changed to `hybrid_remote_train` for better performance

### Bug Fixes
- Fixed `is_configured()` error in `sim.py` (changed to check `wandb.enabled`)
- Fixed incorrect path in `SkypilotDispatcher` for launch script
- Updated checkpoint URI handling for sweeps

## API Reference

See the docstrings in:
- `metta/tools/sweep.py`: Main tool interface
- `metta/sweep/controller.py`: Core controller logic
- `metta/sweep/protein_config.py`: Parameter configuration
- `metta/sweep/scheduler/optimizing.py`: Optimizer-based scheduling
- `metta/sweep/dispatcher/`: Dispatcher implementations
- `metta/rl/loss/`: New loss configuration system