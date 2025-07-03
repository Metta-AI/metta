# Sweep System README

## Overview

The sweep system enables hyperparameter optimization using Protein (Bayesian optimization) with WandB integration. Each sweep consists of multiple training  runs with different hyperparameter configurations.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   sweep_init    │───▶│   train.py      │───▶│  sweep_eval     │
│                 │    │                 │    │                 │
│ • Creates sweep │    │ • Trains model  │    │ • Evaluates     │
│ • Gets run_id   │    │ • Saves policy  │    │ • Records obs   │
│ • Applies       │    │ • Checkpoints   │    │ • Updates       │
│   suggestions   │    │                 │    │   Protein       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     WandB       │    │  Local Storage  │    │    Protein      │
│                 │    │                 │    │                 │
│ • Sweep config  │    │ • Run configs   │    │ • Observations  │
│ • Run tracking  │    │ • Checkpoints   │    │ • Next suggest  │
│ • Metrics       │    │ • Logs          │    │ • Optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

### Sweep Config (`configs/sweep/`)

```yaml
# configs/sweep/quick.yaml
protein:
  num_random_samples: 5
  max_suggestion_cost: 3600
  resample_frequency: 0
  global_search_scale: 1
  random_suggestions: 1024
  suggestions_per_pareto: 256

metric: reward
goal: maximize
method: bayes

parameters:
  trainer:
    optimizer:
      learning_rate:
        distribution: log_normal
        min: 0.0001
        max: 0.001
        mean: 0.0005
        scale: 0.5
```

### Sweep Job Config (`configs/sweep_job.yaml`)

Main configuration that combines:
- `trainer`: Training parameters
- `sim`: Evaluation suite
- `sweep`: Optimization config
- `wandb`: Tracking settings

Key parameters:
- `sweep_run`: Sweep name (e.g., "my_experiment")
- `device`: Training device (cpu/cuda)
- `runs_dir`: Output directory for runs

## Commands

### Local Execution

```bash
# Single sweep run
./devops/sweep_rollout.sh sweep_run=my_experiment +hardware=macbook

# Continuous sweep (multiple runs)
./devops/sweep.sh sweep_run=my_experiment +hardware=macbook

# Custom config overrides
./devops/sweep.sh sweep_run=my_experiment +hardware=macbook trainer.total_timesteps=1000000
```

### Skypilot Execution

**⚠️ Important:** Skypilot currently requires both `run=...` and `sweep_run=...` parameters. This will be fixed in the next PR.

```bash
# Launch sweep on cloud
sky launch skypilot/sweep.yaml --env sweep_run=my_experiment --env run=my_experiment.cloud.001

# With custom hardware
sky launch skypilot/sweep.yaml --env sweep_run=my_experiment --env run=my_experiment.cloud.001 +hardware=aws
```

## File Structure

```
sweep_run_name/
├── config.yaml              # Sweep metadata
├── runs/                     # Individual training runs
│   ├── sweep_name.r.0/      # First run
│   │   ├── train_config_overrides.yaml
│   │   ├── checkpoints/
│   │   └── sweep_eval_results.yaml
│   ├── sweep_name.r.1/      # Second run
│   └── ...
└── dist_*.yaml              # Distributed coordination files
```

## Key Scripts

- `tools/sweep_init.py`: Initialize sweep and create runs
- `tools/train.py`: Train model with suggested hyperparameters
- `tools/sweep_eval.py`: Evaluate trained policy and record results
- `devops/sweep.sh`: Continuous sweep execution
- `devops/sweep_rollout.sh`: Single sweep iteration

## Troubleshooting

**Import errors**: Ensure all packages are installed:
```bash
uv pip install -e ./common -e ./agent -e ./app_backend
```

**Run ID conflicts**: The system automatically generates unique run IDs (e.g., `sweep_name.r.0`, `sweep_name.r.1`)

**WandB issues**: Check that `wandb` config has correct `project` and `entity` settings
