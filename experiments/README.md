# Experiments

A framework for reproducible experiments and research notebooks with Metta.

## Quick Start

### Create and Launch Experiments

```bash
# Create and launch arena experiment
uv run experiments/recipes/arena_experiment.py
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


# TODO:

├── notebooks/            # Notebook utilities
...
├── scratch/              # Generated notebooks (git-ignored)
└── log/                  # HTML exports from notebooks
```

