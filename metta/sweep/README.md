# Metta Sweep System

A hyperparameter optimization system using Protein (Bayesian optimization with Gaussian Processes) integrated with WandB for efficient hyperparameter search and experiment tracking.

## Overview

The sweep system enables automated hyperparameter optimization for training runs. Each sweep consists of multiple training iterations with different hyperparameter configurations, where each iteration:
1. Gets suggestions from the Protein optimizer
2. Trains a model with those hyperparameters
3. Evaluates the trained model
4. Records results back to the optimizer

## Architecture

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

## Quick Start

### Local Execution

```bash
./devops/sweep.sh run=my_experiment +hardware=macbook
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

## Configuration

### Sweep Config (`configs/sweep/`)

```yaml
# configs/sweep/quick.yaml
protein:
  num_random_samples: 5          # Initial random exploration
  max_suggestion_cost: 3600      # Max cost per suggestion (seconds)
  resample_frequency: 0          # How often to resample suggestions
  global_search_scale: 1         # Exploration vs exploitation
  random_suggestions: 1024       # Random samples for acquisition
  suggestions_per_pareto: 256    # Samples per Pareto point

metric: reward                   # Objective metric name
goal: maximize                   # maximize or minimize
method: bayes                    # Optimization method

parameters:
  trainer:
    optimizer:
      learning_rate:
        distribution: log_normal
        min: 0.0001
        max: 0.001
        mean: 0.0005            # Search center point
        scale: 0.5              # Search width
```

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
  distribution: "uniform"
  min: 0.001
  max: 0.01
  scale: "auto"  # or numeric value, controls search width
  mean: 0.005    # search center point
```

### `int_uniform` - Integer uniform distribution
```yaml
batch_size:
  distribution: "int_uniform"
  min: 16
  max: 128
  scale: "auto"
  mean: 64
```

### `log_normal` - Log-normal distribution
Best for parameters that vary over orders of magnitude (learning rates, regularization).
```yaml
learning_rate:
  distribution: "log_normal"
  min: 1e-5
  max: 1e-2
  scale: "auto"  # or "time" for time-based scaling
  mean: 3e-4
```

### `uniform_pow2` - Power-of-2 uniform distribution
For memory-aligned values (batch sizes, hidden dimensions).
```yaml
hidden_size:
  distribution: "uniform_pow2"
  min: 64
  max: 1024
  scale: "auto"
  mean: 256
```

### `logit_normal` - Logit-normal distribution
For probabilities and rates (dropout, clip ratios).
```yaml
dropout_rate:
  distribution: "logit_normal"
  min: 0.1
  max: 0.9
  scale: "auto"
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

## Troubleshooting

### Run ID Conflicts
The system automatically generates unique run IDs (e.g., `sweep_name.r.0`, `sweep_name.r.1`). If conflicts occur, the system will find the next available ID.

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

The `tools/sweep_best_params.py` script helps you extract the best performing hyperparameters from a completed sweep:

```bash
# Basic usage - generates config patch file
./tools/sweep_best_params.py sweep_run=my_sweep_name

# Show top N configurations
./tools/sweep_best_params.py sweep_run=my_sweep_name --top-n 5

# Show all run scores
./tools/sweep_best_params.py sweep_run=my_sweep_name --show-scores

# Custom output directory for patches
./tools/sweep_best_params.py sweep_run=my_sweep_name --output-dir my_patches

# Skip patch generation (only show parameters)
./tools/sweep_best_params.py sweep_run=my_sweep_name --no-patch

# Combine multiple options
./tools/sweep_best_params.py sweep_run=my_sweep_name --top-n 3 --show-scores

# Show help
./tools/sweep_best_params.py --help
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

- `sweep_run=<name>` (required): Name of the sweep to analyze
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
