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
            },
            "model": {
                "dropout_rate": {
                    "distribution": "logit_normal",
                    "min": 0.1,
                    "max": 0.8,
                    "scale": "auto",
                    "mean": 0.3
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
print(f"Try dropout_rate: {suggestion['model']['dropout_rate']}")

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
- Parameter definitions with distributions and required fields:

#### Available Distributions

**`uniform`** - Linear uniform distribution
```yaml
learning_rate:
  distribution: "uniform"
  min: 0.001
  max: 0.01
  scale: "auto"  # or numeric value, controls search width
  mean: 0.005    # search center point
```

**`int_uniform`** - Integer uniform distribution
```yaml
batch_size:
  distribution: "int_uniform"
  min: 16
  max: 128
  scale: "auto"
  mean: 64
```

**`log_normal`** - Log-normal distribution (good for learning rates, regularization)
```yaml
learning_rate:
  distribution: "log_normal"
  min: 1e-5
  max: 1e-2
  scale: "auto"  # or "time" for time-based scaling
  mean: 3e-4
```

**`uniform_pow2`** - Power-of-2 uniform distribution (for memory sizes, batch sizes)
```yaml
hidden_size:
  distribution: "uniform_pow2"
  min: 64
  max: 1024
  scale: "auto"
  mean: 256
```

**`logit_normal`** - Logit-normal distribution (good for probabilities, dropout rates)
```yaml
dropout_rate:
  distribution: "logit_normal"
  min: 0.1
  max: 0.9
  scale: "auto"
  mean: 0.5
```

#### Scale Options
- `"auto"`: Default scale of 0.5
- `"time"`: For log distributions, scale = 1/(log2(max) - log2(min))
- Numeric value: Custom search width around the mean

## Features

- **Gaussian Process optimization** for sample-efficient search
- **WandB integration** for experiment tracking and history
- **OmegaConf support** with interpolation resolution
- **Numpy type conversion** for compatibility
- **Multi-objective optimization** with Pareto frontiers
- **Historical run loading** from WandB sweeps
