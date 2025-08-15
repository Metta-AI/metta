# Metta Sweep System

A hyperparameter optimization system using Protein (Bayesian optimization with Gaussian Processes) integrated with WandB
for efficient hyperparameter search and experiment tracking.

## Overview

The sweep system enables automated hyperparameter optimization for training runs. Each sweep consists of multiple
training iterations with different hyperparameter configurations, where each iteration:

1. Gets suggestions from the Protein optimizer
2. Trains a model with those hyperparameters
3. Evaluates the trained model
4. Records results back to the optimizer

## Architecture

The sweep system follows a modular architecture with clear separation of concerns:

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

### Core Workflow

The sweep execution follows this pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│   sweep_init    │───▶│     train.py    │───▶│   sweep_eval      │
│                 │    │                 │    │                   │
│ • Create sweep  │    │ • Load overrides│    │ • Evaluate        │
│ • Get run_id    │    │ • Train model   │    │ • Record obs      │
│ • Fetch obs     │    │ • Save policy   │    │ • Update Protein/WB│
│ • Gen sugg      │    │ • Checkpoints   │    │                   │
│ • Apply sugg    │    │                 │    │                   │
│   observations  │    │                 │    │                   │
└─────────────────┘    └─────────────────┘    └───────────────────┘
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
./devops/sweep.sh run=my_experiment
```

### Cloud Execution (Skypilot)

```bash
# Launch sweep on cloud
./devops/skypilot/launch.sh sweep run=my_sweep
```

## Components

### Core Modules

- **`protein.py`** - Core Protein optimizer with Gaussian Process models
- **`protein_wandb.py`** - WandB integration for experiment tracking and history
- **`protein_metta.py`** - Metta-specific wrapper with OmegaConf support

### Key Scripts

- **`tools/sweep_init.py`** - Initialize sweep and create runs
- **`tools/train.py`** - Train model with suggested hyperparameters
- **`tools/sweep_eval.py`** - Evaluate trained policy and record results
- **`devops/sweep.sh`** - Continuous sweep execution with retry logic
- **`devops/sweep_rollout.sh`** - Single sweep iteration

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

### Sweep Config (`configs/sweep/`)

````yaml
# configs/sweep/quick.yaml
protein:
  num_random_samples: 5 # Initial random exploration
  max_suggestion_cost: 3600 # Max cost per suggestion (seconds)
  resample_frequency: 0 # How often to resample suggestions
  global_search_scale: 1 # Exploration vs exploitation
  random_suggestions: 1024 # Random samples for acquisition
  suggestions_per_pareto: 256 # Samples per Pareto point

metric: reward # Objective metric name
goal: maximize # maximize or minimize
method: bayes # Optimization method

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
````

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

````

### Sweep Job Config (`configs/sweep_job.yaml`)

Main configuration that combines:

- `trainer`: Training parameters
- `sim`: Evaluation suite
- `sweep`: Optimization config
- `wandb`: Tracking settings

Key parameters:

- `run`: Sweep name (e.g., "my_experiment")
- `runs_dir`: Output directory for runs

## Parameter Distributions

### `uniform` - Linear uniform distribution

```yaml
learning_rate:
  distribution: 'uniform'
  min: 0.001
  max: 0.01
  scale: 'auto' # or numeric value, controls search width
  mean: 0.005 # search center point
````

### `int_uniform` - Integer uniform distribution

```yaml
batch_size:
  distribution: 'int_uniform'
  min: 16
  max: 128
  scale: 'auto'
  mean: 64
```

### `log_normal` - Log-normal distribution

Best for parameters that vary over orders of magnitude (learning rates, regularization).

```yaml
learning_rate:
  distribution: 'log_normal'
  min: 1e-5
  max: 1e-2
  scale: 'auto' # or "time" for time-based scaling
  mean: 3e-4
```

### `uniform_pow2` - Power-of-2 uniform distribution

For memory-aligned values (batch sizes, hidden dimensions).

```yaml
hidden_size:
  distribution: 'uniform_pow2'
  min: 64
  max: 1024
  scale: 'auto'
  mean: 256
```

### `logit_normal` - Logit-normal distribution

For probabilities and rates (dropout, clip ratios).

```yaml
dropout_rate:
  distribution: 'logit_normal'
  min: 0.1
  max: 0.9
  scale: 'auto'
  mean: 0.5
```

### Scale Options

- `"auto"`: Default scale of 0.5
- `"time"`: For log distributions, scale = 1/(log2(max) - log2(min))
- Numeric value: Custom search width around the mean

## File Structure

```
train_dir/sweep/sweep_name/
├── config.yaml              # Sweep metadata & wandb_sweep_id
├── runs/                    # Individual training runs
│   ├── sweep_name.r.0/      # First run
│   │   ├── train_config_overrides.yaml
│   │   ├── checkpoints/
│   │   └── sweep_eval_results.yaml
│   ├── sweep_name.r.1/      # Second run
│   └── ...
└── dist_*.yaml              # Distributed coordination files
```

## Programmatic Usage

```python
from metta.sweep.protein_metta import MettaProtein
from omegaconf import OmegaConf
import wandb

# Setup config
config = OmegaConf.create({
    "sweep": {
        "protein": {
            "max_suggestion_cost": 3600,
            "num_random_samples": 50
        },
        "parameters": {
            "metric": "reward",
            "goal": "maximize",
            "trainer": {
                "optimizer": {
                    "learning_rate": {
                        "distribution": "log_normal",
                        "min": 1e-5,
                        "max": 1e-2,
                        "scale": "auto",
                        "mean": 3e-4
                    }
                },
                "batch_size": {
                    "distribution": "uniform_pow2",
                    "min": 16,
                    "max": 128,
                    "scale": "auto",
                    "mean": 64
                }
            }
        }
    }
})

# Initialize with wandb
wandb.init(project="my_project")
optimizer = MettaProtein(config)

# Get suggestions
suggestion, info = optimizer.suggest()
print(f"Try learning_rate: {suggestion['trainer']['optimizer']['learning_rate']}")
print(f"Try batch_size: {suggestion['trainer']['batch_size']}")

# Train your model...

# Record results
optimizer.record_observation(objective=objective_value, cost=120.0)
```

## Features

- **Gaussian Process optimization** for sample-efficient search
- **WandB integration** for experiment tracking and history
- **Local filesystem caching** for fast sweep ID lookups
- **OmegaConf support** with interpolation resolution
- **Numpy type conversion** for compatibility
- **Multi-objective optimization** with Pareto frontiers
- **Historical run loading** from WandB sweeps
- **Automatic run ID generation** with collision detection
- **Distributed training support** via coordination files

## Performance Optimizations

- **Local cache**: Sweep IDs are cached locally to avoid expensive WandB API searches
- **Batch loading**: Previous runs are loaded in batches for efficiency
- **Lazy evaluation**: Suggestions are only computed when needed

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

### Run ID Conflicts

The system automatically generates unique run IDs (e.g., `sweep_name.r.0`, `sweep_name.r.1`). If conflicts occur, the
system will find the next available ID.

### WandB Issues

- Check that `wandb` config has correct `project` and `entity` settings
- Ensure you're logged in: `wandb login`
- Verify sweep exists: Check the cached sweep ID in `train_dir/sweep/{sweep_name}/config.yaml`

## Development

### Running Tests

```bash
# Run all sweep tests
cd tests && python -m pytest sweep/ -xvs

# Run specific test file
python -m pytest sweep/test_protein_metta.py -xvs
```

### Adding New Distributions

To add a new parameter distribution:

1. Implement the distribution in `protein.py`
2. Add support in `_process_parameter_config` in `protein_metta.py`
3. Update this README with the new distribution
4. Add tests in `tests/sweep/`

## Analyzing Sweep Results

### Extracting Best Parameters

The `tools/get_best_params_from_sweep.py` script helps you extract the best performing hyperparameters from a completed
sweep:

```bash
# Basic usage - generates config patch file
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name

# Show top N configurations
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name --top-n 5

# Show all run scores
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name --show-scores

# Custom output directory for patches
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name --output-dir my_patches

# Skip patch generation (only show parameters)
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name --no-patch

# Combine multiple options
./tools/get_best_params_from_sweep.py sweep_name=my_sweep_name --top-n 3 --show-scores

# Show help
./tools/get_best_params_from_sweep.py --help
```

#### Features

- **Automatic sweep discovery**: Finds sweep by name in your WandB project
- **Config patch generation**: Creates Hydra-compatible patch files with `@package _global_` directive
- **Multiple output formats**:
  - YAML config patch for use with `+trainer/patch=`
  - Command-line overrides for direct use
  - Complete training commands
- **Scientific notation**: Small values (< 0.01) are formatted in scientific notation for readability
- **Top-N analysis**: Compare multiple high-performing configurations
- **Score display**: Optionally show scores for all successful runs

#### Available Options

- `sweep_name=<name>` (required): Name of the sweep to analyze
- `--top-n <int>`: Number of top runs to analyze (default: 1)
- `--show-scores`: Show scores for all runs
- `--no-patch`: Skip generating config patch file
- `--output-dir <path>`: Directory for patch files (default: configs/trainer/patch)

#### Generated Outputs

The script generates multiple formats for using the best parameters:

1. **Config Patch File** (saved to `configs/trainer/patch/{sweep_name}_best.yaml`):

```yaml
# @package _global_
# Best hyperparameters from sweep
# Apply with: +trainer/patch=<filename_without_yaml>

trainer:
  optimizer:
    learning_rate: 7.31e-04
```

#### Using the Config Patch

The generated patch file can be used with Hydra's composition system:

```bash
# Train with best parameters from sweep
./devops/train.sh run=new_experiment +trainer/patch=my_sweep_name_best

# The patch will override the default trainer configuration
# Additional overrides can still be applied
./devops/train.sh run=new_experiment +trainer/patch=my_sweep_name_best trainer.batch_size=128
```

#### Example Output

```
Best run: my_sweep.r.42 (score: 0.8734)

============================================================
BEST HYPERPARAMETERS:
============================================================

1. As YAML config:
----------------------------------------
trainer:
  optimizer:
    learning_rate: 7.31e-04

2. Config patch saved to: configs/trainer/patch/my_sweep_best.yaml
   Use with: +trainer/patch=my_sweep_best

3. As command-line overrides:
----------------------------------------
./devops/train.sh trainer.optimizer.learning_rate=7.31e-04

4. Complete training command:
----------------------------------------
./devops/train.sh run=my_sweep_best trainer.optimizer.learning_rate=7.31e-04
```
