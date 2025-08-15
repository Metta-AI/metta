# Metta Sweep System

## Overview

NOTE: This README needs to be combed through carefully. The Metta sweep system provides automated hyperparameter
optimization using Bayesian optimization through the Protein optimizer, integrated with Weights & Biases (WandB) for
experiment tracking and a centralized Cogweb database for sweep coordination. The system efficiently explores
hyperparameter spaces to find optimal training configurations for reinforcement learning policies.

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
experiments/
└── dashboards/
    └── sweep_dashboard.py                # Interactive analysis dashboard
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

### Sweep Phasing System

The sweep system supports a phasing mechanism that allows you to dynamically adjust hyperparameter search strategies as
the sweep progresses. This enables efficient exploration-exploitation trade-offs by starting with cheap exploratory runs
and progressively moving to more expensive, focused exploitation.

#### How Phasing Works

1. **Phase Progression**: Phases are defined in the `schedule.phases` list and progress based on the total number of
   completed runs
2. **Dynamic Configuration**: Each phase can override any sweep configuration parameters (Protein settings, parameter
   distributions, etc.)
3. **Automatic Transition**: The system automatically switches to the next phase when the run count threshold is reached
4. **Configuration Merging**: Phase-specific settings are merged with the base sweep configuration

#### Example Phased Sweep Configuration

```yaml
schedule:
  phases:
    - name: 'explore_cheap'
      num_runs: 100 # First 100 runs use this phase
      sweep:
        protein:
          max_suggestion_cost: 1800 # 30 minutes max per run
          random_suggestions: 18 # More exploration
          expansion_rate: 0.18 # Higher exploration rate
        parameters:
          trainer:
            total_timesteps: { mean: 300000000 } # 300M steps
            batch_size: { mean: 524288 } # 2^19
            minibatch_size: { mean: 8192 } # 2^13

    - name: 'exploit_medium'
      num_runs: 70 # Runs 101-170 use this phase
      sweep:
        protein:
          max_suggestion_cost: 3600 # 1 hour max per run
          random_suggestions: 10 # Less exploration
          suggestions_per_pareto: 24 # More focused sampling
          expansion_rate: 0.10 # Lower exploration rate
        parameters:
          trainer:
            total_timesteps: { mean: 750000000 } # 750M steps
            batch_size: { mean: 1048576 } # 2^20
            minibatch_size: { mean: 16384 } # 2^14

    - name: 'peak_expensive'
      num_runs: 11 # Runs 171+ use this phase
      sweep:
        protein:
          max_suggestion_cost: 7200 # 2 hours max per run
          random_suggestions: 5 # Minimal exploration
          suggestions_per_pareto: 16 # Highly focused
          expansion_rate: 0.06 # Very low exploration
        parameters:
          trainer:
            total_timesteps: { mean: 1000000000 } # 1B steps
            batch_size: { mean: 1048576 } # 2^20
            minibatch_size: { mean: 16384 } # 2^14
```

#### Phase Configuration Options

Each phase can override:

1. **Protein Settings** (`sweep.protein.*`):
   - `max_suggestion_cost`: Maximum compute time per run (in seconds)
   - `random_suggestions`: Number of random samples for acquisition
   - `suggestions_per_pareto`: Samples per Pareto point
   - `expansion_rate`: Exploration along cost dimension
   - `resample_frequency`: Frequency of resampling from Pareto front
   - `num_random_samples`: Initial random exploration samples
   - `global_search_scale`: Exploration vs exploitation balance

2. **Parameter Distributions** (`sweep.parameters.*`):
   - Adjust `mean` values to shift search centers
   - Modify `min`/`max` bounds to narrow/widen search
   - Change `scale` for exploration width
   - Override entire distribution configurations

#### Benefits of Phasing

1. **Cost Efficiency**: Start with cheap runs to identify promising regions
2. **Progressive Refinement**: Gradually increase run quality and focus
3. **Adaptive Exploration**: Adjust exploration-exploitation balance over time
4. **Resource Optimization**: Allocate more compute to promising configurations
5. **Risk Management**: Avoid expensive failures in early exploration

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
   - **Phase Selection** (if `schedule` is configured):
     - Counts total completed runs from observations
     - Determines current phase based on `num_runs` thresholds
     - Merges phase-specific overrides with base sweep config
     - Creates phase-specific Protein optimizer instance
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

### Interactive Sweep Dashboard

The sweep system includes a powerful interactive dashboard for real-time analysis and visualization of sweep results.
The dashboard provides WandB-quality visualizations with full interactivity.

#### Running the Dashboard

```bash
# Basic usage
python experiments/dashboards/sweep_dashboard.py --sweep-name my_sweep

# With custom entity and project
python experiments/dashboards/sweep_dashboard.py \
  --sweep-name my_sweep \
  --entity metta-research \
  --project metta

# Limit observations and set hourly cost
python experiments/dashboards/sweep_dashboard.py \
  --sweep-name my_sweep \
  --max-observations 500 \
  --hourly-cost 5.0
```

#### Dashboard Features

The interactive dashboard provides:

1. **Summary Statistics Cards**:
   - Total runs completed
   - Best score achieved
   - Total compute cost
   - Average runtime

2. **Interactive Visualizations**:
   - **Cost vs Score Analysis**: Scatter plot with Pareto frontier highlighting
   - **Parameter Importance**: Bar chart showing correlations with objective
   - **Score Progression**: Timeline view with moving average
   - **Efficiency Frontier**: Pareto optimal runs visualization
   - **Distributions**: Histograms for score and cost distributions
   - **Parameter Correlations**: Grid of scatter plots for all parameters vs score

3. **Interactive Features**:
   - **Dynamic Filtering**: Adjust score and cost ranges with sliders
   - **Click for Details**: Click any point to see full run configuration
   - **Hover Information**: Detailed tooltips on all visualizations
   - **Trend Lines**: Automatic trend fitting for parameter correlations

4. **Run Details Panel**:
   - Click on any data point to display:
     - Complete hyperparameter configuration
     - Performance metrics
     - Runtime and cost information
     - Run identification details

#### Dashboard Access

Once launched, the dashboard runs as a local web server:

- URL: `http://127.0.0.1:8050/`
- Real-time updates as new runs complete
- Export capabilities for all visualizations

#### Dashboard Requirements

The dashboard requires the following Python packages (typically already installed):

- `dash` and `dash-bootstrap-components` for the web interface
- `plotly` for interactive visualizations
- `pandas` and `numpy` for data processing
- `wandb` for fetching sweep data

Install if needed:

```bash
pip install dash dash-bootstrap-components plotly
```

### Extract Best Parameters

```bash
# Get best configuration from sweep
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

### Monitor Progress in WandB

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

### General Guidelines

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

### Phasing Best Practices

1. **Phase Design**:
   - Start with 50-100 cheap exploration runs
   - Use 30-70 medium runs for refinement
   - Reserve 10-20 expensive runs for final optimization
   - Total phase runs should match your compute budget

2. **Cost Progression**:
   - Each phase should be 2-4x more expensive than previous
   - Align `max_suggestion_cost` with actual expected runtime (in seconds)
   - Gradually increase `max_suggestion_cost` across phases

3. **Parameter Tuning**:
   - Gradually increase `total_timesteps` across phases
   - Start with smaller batch sizes for faster iteration
   - Shift `mean` values based on promising regions found

4. **Exploration Strategy**:
   - High `random_suggestions` (15-20) in early phases
   - Low `random_suggestions` (3-5) in final phases
   - Decrease `expansion_rate` as phases progress
   - Increase `suggestions_per_pareto` for exploitation phases

5. **Monitoring Transitions**:
   - Watch WandB for phase transitions in run names
   - Verify cost/performance trade-offs align with expectations
   - Check that Protein is converging before final phase

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
