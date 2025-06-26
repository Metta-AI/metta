# Protein Hyperparameter Optimization

A Bayesian hyperparameter optimization system using Gaussian Processes for efficient hyperparameter search with WandB integration.

## Components

- **`protein.py`** - Core Protein optimizer with Gaussian Process models
- **`protein_wandb.py`** - WandB integration for experiment tracking
- **`protein_metta.py`** - Metta-specific wrapper with OmegaConf support

## Usage

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
            "learning_rate": {
                "distribution": "log_normal",
                "min": 1e-5,
                "max": 1e-2,
                "scale": "auto",
                "mean": 3e-4
            }
        }
    }
})

# Initialize with wandb
wandb.init(project="my_project")
optimizer = MettaProtein(config)

# Get suggestions
suggestion, info = optimizer.suggest()
print(f"Try learning_rate: {suggestion['learning_rate']}")

# Record results
optimizer.record_observation(objective=0.95, cost=120.0)
```

## Configuration

### Protein Settings (`sweep.protein`)
- `max_suggestion_cost`: Maximum cost per suggestion
- `num_random_samples`: Initial random samples
- `global_search_scale`: Search exploration scale

### Parameters (`sweep.parameters`)
- `metric`: Objective metric name
- `goal`: "maximize" or "minimize"
- Parameter definitions with distributions:
  - `log_normal`: Log-normal distribution
  - `uniform`: Uniform distribution
  - Each with `min`, `max`, `scale`, `mean`

## Features

- **Gaussian Process optimization** for sample-efficient search
- **WandB integration** for experiment tracking and history
- **OmegaConf support** with interpolation resolution
- **Numpy type conversion** for compatibility
- **Multi-objective optimization** with Pareto frontiers
- **Historical run loading** from WandB sweeps
