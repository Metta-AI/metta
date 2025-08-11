# Experiments

A framework for reproducible experiments with Metta. This includes:
- Systematic launching of training jobs for MARL research experiments
- Notebook infrastructure for analysis and visualization (marimo-based)

Note: The experiment launching and notebook systems are not yet connected. Integration will be added in a future update.

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
├── training_job.py       # TrainingJob and TrainingJobConfig
├── wandb_service.py      # WandB data fetching utilities  
├── skypilot_service.py   # Sky job monitoring utilities
├── recipes/              # Experiment implementations
│   └── arena_experiment.py
├── notebooks/            # Notebook infrastructure (marimo/jupyter)
├── marimo/              # Marimo-based notebooks
└── scratch/              # Working directory (git-ignored)
```

