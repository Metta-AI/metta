# Experiments

A framework for reproducible experiments.

## Quick Start

### Create and Launch Experiments

```bash
# Create and launch arena experiment
./experiments/recipes/arena_experiment.py

# Create notebook without launching (load existing jobs)
./experiments/recipes/arena_experiment.py --no-launch --job-ids 2979 2980 --open

# Launch with custom configuration
./experiments/recipes/arena_experiment.py my_arena --gpus 4 --skip-git-check --wandb-tags research ablation

```

## Structure

```
experiments/
├── experiment.py          # Base Experiment class
├── launch.py             # Core training launch functionality
├── types.py              # TrainingJob and TrainingJobConfig
├── wandb_utils.py        # WandB data fetching utilities
├── monitoring.py         # Sky job monitoring utilities
├── recipes/              # Experiment implementations
│   └── arena_experiment.py
│   └── bbc_experiment.py

# TODO:

├── notebooks/            # Notebook utilities
├── scratch/              # Generated notebooks (git-ignored)
└── log/                  # HTML exports from notebooks
```
