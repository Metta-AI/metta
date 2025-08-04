# Metta Sweep System

## Overview

The Metta sweep system provides automated hyperparameter optimization using Bayesian optimization through the Protein
optimizer, integrated with Weights & Biases (WandB) for experiment tracking and a centralized Cogweb database for sweep
coordination. The system efficiently explores hyperparameter spaces to find optimal training configurations for
reinforcement learning policies.

## System Architecture

### Core Components

```
metta/
├── sweep/                                 # Core sweep modules
│   ├── __init__.py                       # Package exports
│   ├── protein.py                        # Gaussian Process Bayesian optimizer
│   ├── protein_metta.py                  # Metta wrapper with OmegaConf support
│   ├── sweep_lifecycle.py                # Sweep initialization, preparation, evaluation
│   └── wandb_utils.py                    # WandB observation management
│
tools/
├── sweep_execute.py                      # Main sweep execution script
├── train.py                              # Training script invoked by sweeps
└── get_best_params_from_sweep.py         # Extract optimal parameters from completed sweeps
│
configs/
├── sweep_job.yaml                        # Main sweep job configuration
├── sweep/                                # Sweep parameter configurations
│   ├── quick.yaml                        # Minimal sweep for testing (5 samples)
│   ├── full.yaml                         # Comprehensive parameter search
│   └── cogeval_sweep.yaml                # Cognitive evaluation sweep config
│
devops/
├── sweep.sh                              # Shell wrapper for sweep execution
└── skypilot/
    └── launch.py                         # Cloud deployment via SkyPilot
```

## Configuration System

### Main Sweep Job Configuration (`configs/sweep_job.yaml`)

```yaml
defaults:
  - common # Common settings (data_dir, device, etc.)
  - wandb: metta_research # WandB configuration
  - sweep: full # Sweep parameter space definition
  - _self_

sweep_name: ??? # Required: set via command line

settings:
  max_consecutive_failures: 0 # 0 = unlimited retries
  rollout_retry_delay: 5 # Seconds between retry attempts
  max_observations_to_load: 250 # Limit historical observations
  sweep_server_uri: https://api.observatory.softmax-research.net

sim_name: arena # Evaluation suite to use

sweep_job_overrides: # Applied to all training runs
  trainer:
    curriculum: /env/mettagrid/arena/advanced
    simulation:
      evaluate_interval: 0 # No evaluation during training
      replay_dir: '' # No replays for sweeps
```

### Sweep Parameter Configurations (`configs/sweep/*.yaml`)

The sweep system supports multiple parameter distributions for different types of hyperparameters:

#### Distribution Types

1. **`uniform`** - Linear uniform distribution

   ```yaml
   parameter:
     distribution: uniform
     min: 0.1 # Lower bound
     max: 1.0 # Upper bound
     mean: 0.5 # Search center
     scale: auto # Search width (auto = 0.5)
   ```

2. **`int_uniform`** - Integer uniform distribution

   ```yaml
   num_layers:
     distribution: int_uniform
     min: 2
     max: 8
     mean: 4
     scale: auto
   ```

3. **`uniform_pow2`** - Power-of-2 integers (for batch sizes, etc.)

   ```yaml
   batch_size:
     distribution: uniform_pow2
     min: 256 # 2^8
     max: 4096 # 2^12
     mean: 1024 # 2^10
     scale: auto
   ```

4. **`log_normal`** - Log-scale distribution (for learning rates, etc.)

   ```yaml
   learning_rate:
     distribution: log_normal
     min: 1e-5
     max: 1e-2
     mean: 3e-4 # Geometric center
     scale: auto # Or "time" for adaptive scaling
   ```

5. **`logit_normal`** - For probabilities (0-1 range)
   ```yaml
   dropout_rate:
     distribution: logit_normal
     min: 0.1
     max: 0.9
     mean: 0.5
     scale: auto
   ```

#### Example Full Sweep Configuration

```yaml
protein: # Protein optimizer settings
  num_random_samples: 20 # Initial exploration samples
  max_suggestion_cost: 3600 # Max compute time per run (seconds)
  resample_frequency: 3 # Frequency of resampling
  global_search_scale: 1.0 # Exploration vs exploitation
  random_suggestions: 15 # Random samples for acquisition
  suggestions_per_pareto: 32 # Samples per Pareto point
  expansion_rate: 0.15 # Exploration along cost dimension
  seed_with_search_center: true # Start from mean values

metric: reward # Objective to optimize
goal: maximize # maximize or minimize
method: bayes # Optimization method

parameters: # Hyperparameter search space
  trainer:
    total_timesteps:
      distribution: int_uniform
      min: 500000000 # 500M steps
      max: 2000000000 # 2B steps
      mean: 1000000000 # 1B steps
      scale: auto

    batch_size:
      distribution: uniform_pow2
      min: 262144 # 2^18
      max: 2097152 # 2^21
      mean: 524288 # 2^19
      scale: auto

    optimizer:
      learning_rate:
        distribution: log_normal
        min: 1e-4
        max: 1e-2
        mean: 3e-4
        scale: auto
```

## Running Sweeps

### Local Execution

```bash
# Basic sweep execution (runs continuously)
./tools/sweep_execute.py sweep_name=my_sweep

# With specific sweep configuration
./tools/sweep_execute.py sweep_name=my_sweep sweep=quick

# Override specific parameters
./tools/sweep_execute.py sweep_name=my_sweep \
  sweep.protein.num_random_samples=10 \
  sweep_job_overrides.trainer.total_timesteps=10000000

# Using the shell wrapper
./devops/sweep.sh run=my_sweep
```

### Cloud Execution (SkyPilot)

```bash
# Launch sweep on cloud
./devops/skypilot/launch.py sweep run=my_sweep

# With specific hardware
./devops/skypilot/launch.py sweep run=my_sweep \
  --gpus 8 \
  --nodes 4

# Using no-spot instances
./devops/skypilot/launch.py sweep run=my_sweep \
  --no-spot
```

## Detailed Execution Flow

### Phase 1: Sweep Initialization (`initialize_sweep`)

1. **Cogweb Registration**
   - Connects to centralized sweep server (`sweep_server_uri`)
   - Checks if sweep exists in database
   - Creates new sweep entry if needed
   - Stores sweep metadata (name, WandB project/entity)

### Phase 2: Continuous Rollout Loop

The main script (`sweep_execute.py`) runs an infinite loop that:

1. **Prepares Each Run** (`prepare_sweep_run`)
   - Fetches previous observations from WandB (up to `max_observations_to_load`)
   - Updates Protein optimizer with historical data
   - Generates new hyperparameter suggestions using Gaussian Process
   - Gets unique run ID from Cogweb (e.g., `sweep_name.r.0`, `sweep_name.r.1`)
   - Returns run name and parameter suggestions

2. **Launches Training** (`launch_training_subprocess`)
   - Constructs command for `devops/train.sh`:
     ```bash
     ./devops/train.sh \
       run=sweep_name.r.0 \
       wandb.entity=<entity> \
       wandb.project=<project> \
       wandb.group=<sweep_name> \
       wandb.name=sweep_name.r.0 \
       ++trainer.batch_size=1024 \           # From protein suggestion
       ++trainer.optimizer.learning_rate=0.0003 \
       ++trainer.total_timesteps=1000000000 \
       sim=arena                              # From sweep_job config
     ```
   - Parameters are passed via CLI using `++` prefix for force-override
   - Training runs as subprocess with real-time logging
   - Outputs saved to `{data_dir}/{run_name}/`

3. **Evaluates Results** (`evaluate_sweep_rollout`)
   - Loads saved configuration from `sweep_eval_config.yaml`
   - Initializes WandB context for recording
   - Runs policy evaluation suite:
     - Loads trained policy from checkpoints
     - Executes simulation suite specified by `sim_name`
     - Collects metrics (reward, success rate, etc.)
   - Records observation to Protein via WandB:
     - Suggestion parameters
     - Objective value (metric score)
     - Compute cost (training + eval time)
     - Failure status
   - Saves results to `sweep_eval_results.yaml`

4. **Error Handling**
   - On failure: increments consecutive failure counter
   - Waits `rollout_retry_delay` seconds before retry
   - Stops if `max_consecutive_failures` exceeded (0 = unlimited)
   - On success: resets failure counter

### Phase 3: Optimization Process

The Protein optimizer uses Gaussian Process regression to:

1. **Model the objective function** from all observations
2. **Balance exploration vs exploitation** via acquisition functions
3. **Consider compute cost** in multi-objective optimization
4. **Suggest promising hyperparameters** for next iteration

Key Protein settings:

- `num_random_samples`: Initial random exploration before GP modeling
- `max_suggestion_cost`: Upper bound on compute time per run
- `global_search_scale`: Controls exploration (higher = more exploration)
- `suggestions_per_pareto`: Samples for Pareto frontier construction
- `seed_with_search_center`: Whether to start from mean values

### Distributed Coordination

For multi-node training:

- **Rank 0 (Master)**: Handles all sweep operations
- **Other ranks**: Wait via `run_once` synchronization
- All ranks participate in distributed training
- Only rank 0 performs evaluation and recording

## Analyzing Results

### Extract Best Parameters

```bash
# Get best configuration
./tools/get_best_params_from_sweep.py sweep_name=my_sweep

# Show top 5 configurations
./tools/get_best_params_from_sweep.py sweep_name=my_sweep --top-n 5

# Generate reusable config patch
./tools/get_best_params_from_sweep.py sweep_name=my_sweep \
  --output-dir configs/trainer/patch
```

This generates a patch file for future training:

```bash
# Train with optimal parameters
./tools/train.py run=production +trainer/patch=my_sweep_best
```

### Monitor Progress

View in WandB dashboard:

- Filter by group name (sweep name)
- Use parallel coordinates plot for parameter relationships
- Track metric convergence over iterations
- Analyze cost vs performance trade-offs

## Integration Points

### Cogweb Server

- Centralized sweep coordination
- Unique run ID generation
- Prevents duplicate runs across workers
- API endpoint: `sweep_server_uri`

### WandB

- Stores all observations and metrics
- Provides sweep visualization
- Enables historical data loading
- Groups runs by sweep name

### Evaluation System

- Uses `SimulationSuite` for policy evaluation
- Runs suite specified by `sim_name`
- Stores results in SQLite database (`eval/stats.db`)
- Computes aggregate metrics for optimization

## Best Practices

1. **Start with `sweep=quick`** for testing (5 samples)
2. **Set appropriate bounds** - avoid extremely wide search spaces
3. **Use correct distributions**:
   - `log_normal` for learning rates
   - `uniform_pow2` for batch sizes
   - `logit_normal` for probabilities
4. **Monitor early iterations** before scaling up
5. **Set `max_consecutive_failures`** > 0 for production
6. **Limit `max_observations_to_load`** for large sweeps
7. **Use descriptive sweep names** for organization

## Troubleshooting

### Common Issues

1. **Import errors**: Check PyTorch installation (macOS issues common)
2. **Cogweb connection**: Verify `sweep_server_uri` is accessible
3. **WandB authentication**: Run `wandb login`
4. **Duplicate runs**: Check Cogweb database for conflicts
5. **OOM errors**: Reduce `batch_size` or `num_envs`

### Debug Mode

```bash
# Enable detailed logging
HYDRA_FULL_ERROR=1 ./tools/sweep_execute.py sweep_name=debug

# Check intermediate files
ls train_dir/<sweep_name>/<run_name>/
```

### Recovery

The system automatically handles:

- Resumption from interruption (no state lost)
- Retry of failed runs
- Loading historical observations
- Skipping duplicate run IDs
