# Sweep Orchestrator Guide

## Overview

The new sweep orchestrator provides a stateless, distributed-friendly hyperparameter optimization system for Metta. It
uses Protein (Bayesian optimization) to efficiently explore hyperparameter spaces and WandB for persistent state
management.

## Quick Start

### Running a Quick Test Sweep (Local)

```bash
# Run a quick test sweep with 5 trials locally
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    sweep_name=my_test_sweep max_trials=5 \
    dispatcher_type=local
```

### Running with Different Recipes

```bash
# Use arena_basic_easy_shaped recipe
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
    sweep_name=test_arena_basic \
    recipe_module=experiments.recipes.arena_basic_easy_shaped \
    train_entrypoint=train\
    eval_entrypoint=evaluate \
    max_trials=5
```

### Running a Full PPO Hyperparameter Sweep (Hybrid Mode)

```bash
# Run a full PPO sweep with 10 trials (default: train on cloud, eval locally)
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    sweep_name=ppo_sweep_001 max_trials=10
```

## Step-by-Step Sweep Workflow Example

Let's walk through a complete PPO sweep using a custom recipe (`arena_basic_easy_shaped`) with specific entrypoints.
This example shows exactly how commands are built and what gets executed.

### Step 1: Launch the Sweep

```bash
uv run ./tools/run.py experiments.sweeps.standard.ppo \
    sweep_name=ppo_arena_basic \
    recipe_module=experiments.recipes.arena_basic_easy_shaped \
    train_entrypoint=train \
    eval_entrypoint=evaluate \
    max_trials=3 \
    dispatcher_type=hybrid_remote_train
```

### Step 2: What Happens Behind the Scenes

#### 2.1 Sweep Initialization

The orchestrator starts and creates the sweep configuration:

```
[SweepOrchestrator] Starting sweep: ppo_arena_basic
[SweepOrchestrator] Recipe: experiments.recipes.arena_basic_easy_shaped.train
[SweepOrchestrator] Max trials: 3
[SweepOrchestrator] Dispatcher type: hybrid_remote_train
```

#### 2.2 First Trial - Training Job

The Protein optimizer suggests hyperparameters, and the orchestrator builds the training command:

```
[BatchedSyncedScheduler] 🚀 Scheduling trial 1/3: trial_0001
[BatchedSyncedScheduler]    trainer.optimizer.learning_rate: 0.0003421
[BatchedSyncedScheduler]    trainer.losses.loss_configs.ppo.clip_coef: 0.182
[BatchedSyncedScheduler]    trainer.losses.loss_configs.ppo.ent_coef: 0.0023
```

**Actual dispatched command (via SkypilotDispatcher for training):**

```bash
/Users/axel/Documents/Softmax/metta-repo/devops/skypilot/launch.py \
    --no-spot \
    --gpus=1 \
    experiments.recipes.arena_basic_easy_shaped.train \
    run=ppo_arena_basic_trial_0001 \
    trainer.optimizer.learning_rate=0.0003421 \
    trainer.losses.loss_configs.ppo.clip_coef=0.182 \
    trainer.losses.loss_configs.ppo.ent_coef=0.0023 \
    trainer.losses.loss_configs.ppo.gae_lambda=0.91 \
    trainer.losses.loss_configs.ppo.vf_coef=0.43
```

This launches the training job on cloud resources (Skypilot).

#### 2.3 Training Progress Monitoring

```
[trial_0001] Epoch 1 [ppo_arena_basic_trial_0001] / 521 sps / 1.31% of 50.00 ksteps
[trial_0001] Epoch 2 [ppo_arena_basic_trial_0001] / 743 sps / 2.62% of 50.00 ksteps
...
[trial_0001] Epoch 50 [ppo_arena_basic_trial_0001] / 817 sps / 100.00% of 50.00 ksteps
[trial_0001] Training complete!
```

#### 2.4 First Trial - Evaluation Job

After training completes, the orchestrator schedules evaluation:

```
[BatchedSyncedScheduler] Scheduling evaluation for trial_0001
```

**Actual dispatched command (via LocalDispatcher for evaluation):**

```bash
uv run ./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate \
    policy_uri=file://./train_dir/ppo_arena_basic_trial_0001/checkpoints/ppo_arena_basic_trial_0001:v50.pt \
    push_metrics_to_wandb=True
```

This command is executed locally (not sent to Skypilot). The evaluation command then submits the evaluation task to a
remote evaluation server (separate from Skypilot).

#### 2.5 Evaluation Results

```
[trial_0001] Running evaluation suite...
[trial_0001] eval_arena: score=0.721, survival_rate=0.89
[BatchedSyncedScheduler] Trial trial_0001 completed with score: 0.721
```

### Step 3: Subsequent Trials

The Protein optimizer uses Bayesian optimization to suggest better hyperparameters based on trial 1's results:

```
[BatchedSyncedScheduler] 🚀 Scheduling trial 2/3: trial_0002
[BatchedSyncedScheduler]    trainer.optimizer.learning_rate: 0.0008124  # Adjusted based on trial 1
[BatchedSyncedScheduler]    trainer.losses.loss_configs.ppo.clip_coef: 0.095  # Adjusted
```

**Trial 2 Training Command:**

```bash
/Users/axel/Documents/Softmax/metta-repo/devops/skypilot/launch.py \
    --no-spot \
    --gpus=1 \
    experiments.recipes.arena_basic_easy_shaped.train \
    run=ppo_arena_basic_trial_0002 \
    trainer.optimizer.learning_rate=0.0008124 \
    trainer.losses.loss_configs.ppo.clip_coef=0.095 \
    # ... other optimized parameters
```

### Step 4: Final Summary

After all trials complete:

```
[SweepOrchestrator] SWEEP SUMMARY
[SweepOrchestrator] ========================================
[SweepOrchestrator] Run ID                    Status         Score
[SweepOrchestrator] ----------------------------------------
[SweepOrchestrator] trial_0001                completed      0.721
[SweepOrchestrator] trial_0002                completed      0.843
[SweepOrchestrator] trial_0003                completed      0.798
[SweepOrchestrator] ========================================
[SweepOrchestrator] Best result:
[SweepOrchestrator]    Run: ppo_arena_basic_trial_0002
[SweepOrchestrator]    Score: 0.843
[SweepOrchestrator]    Config: {
    "trainer.optimizer.learning_rate": 0.0008124,
    "trainer.losses.loss_configs.ppo.clip_coef": 0.095,
    "trainer.losses.loss_configs.ppo.ent_coef": 0.0041,
    "trainer.losses.loss_configs.ppo.gae_lambda": 0.96,
    "trainer.losses.loss_configs.ppo.vf_coef": 0.28
}
```

### Understanding the Command Flow

1. **User Command** → Calls `experiments.sweeps.standard.ppo` function
2. **Sweep Tool** → Creates `SweepTool` with PPO parameter search space
3. **Orchestrator** → Manages the overall sweep process
4. **Scheduler** → Uses Protein optimizer to suggest hyperparameters
5. **Dispatcher** → Routes jobs:
   - Training → Sent to Skypilot (executes on cloud GPUs)
   - Evaluation → Executed locally (but submits to remote eval server)
6. **Actual Commands** → Fully expanded with all parameters and overrides

### Key Points

- **Hybrid dispatch**: Training commands go through Skypilot to cloud GPUs, evaluation commands execute locally but
  submit tasks to remote evaluation servers
- **Parameter paths**: Note the full paths like `trainer.losses.loss_configs.ppo.clip_coef`
- **Bayesian optimization**: Each trial learns from previous results
- **Automatic checkpointing**: Training saves checkpoints that evaluation loads
- **WandB tracking**: All metrics are logged to WandB for visualization

## Dispatcher Types

The sweep system supports three dispatcher modes:

- **`local`**: Commands are executed locally (but may still dispatch to remote infrastructure)
- **`skypilot`**: Commands are sent to Skypilot for cloud execution
- **`hybrid_remote_train`** (default): Training commands sent to Skypilot, evaluation commands executed locally

```bash
# Force local execution
dispatcher_type=local

# Force cloud execution
dispatcher_type=skypilot

# Use hybrid mode (default)
dispatcher_type=hybrid_remote_train
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
   - `BatchedSyncedOptimizingScheduler`: Uses Protein optimizer for Bayesian optimization with batched synchronous
     execution
3. **Store** (`metta/sweep/store/wandb.py`): Persistent state management via WandB
4. **Dispatcher**: Executes jobs
   - `LocalDispatcher`: Local subprocess execution with output capture
   - `SkypilotDispatcher`: Cloud execution via Skypilot
   - `RoutingDispatcher`: Routes different job types to different dispatchers

## Command-Line Arguments

The sweep tool now accepts these arguments:

- `sweep_name`: Name of the sweep (auto-generated if not provided)
- `max_trials`: Maximum number of trials to run
- `recipe_module`: Recipe module to use (e.g., `experiments.recipes.arena`)
- `train_entrypoint`: Training function name (e.g., `train_shaped`)
- `eval_entrypoint`: Evaluation function name (e.g., `evaluate`)

Example with all arguments:

```bash
uv run ./tools/run.py experiments.sweeps.standard.quick_test \
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
from metta.tools.sweep import SweepTool

def my_custom_sweep(
    sweep_name: str = None,
    max_trials: int = 20,
) -> SweepTool:
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

    return SweepTool(
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
[BatchedSyncedScheduler] 🚀 Scheduling trial 1/10: trial_0001
[BatchedSyncedScheduler]    trainer.optimizer.learning_rate: 0.001
[BatchedSyncedScheduler]    trainer.losses.loss_configs.ppo.clip_coef: 0.15
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
SweepTool(
    max_parallel_jobs=4,  # Run 4 jobs in parallel
    monitoring_interval=10,  # Check job status every 10 seconds
)
```

### Custom Training Overrides

```python
SweepTool(
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
    sweep_name=ppo_sweep_001 max_trials=10
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
dispatcher_type=local capture_output=true
```

## Examples in the Codebase

- `experiments/sweeps/standard.py`: Standard sweep configurations
  - `ppo()`: Full PPO hyperparameter sweep
  - `quick_test()`: Quick test configuration with fewer trials

## API Reference

See the docstrings in:

- `metta/tools/sweep.py`: Main tool interface
- `metta/sweep/controller.py`: Core controller logic
- `metta/sweep/protein_config.py`: Parameter configuration
- `metta/sweep/scheduler/optimizing.py`: Optimizer-based scheduling
- `metta/sweep/dispatcher/`: Dispatcher implementations
- `metta/rl/loss/`: New loss configuration system
