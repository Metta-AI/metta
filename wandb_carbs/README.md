# Wandb Carbs

[![PyPI version](https://badge.fury.io/py/wandb-carbs.svg)](https://badge.fury.io/py/wandb-carbs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Wandb Carbs is a Python package that integrates the [CARBS](https://github.com/imbue-ai/carbs) (Cost-Aware Bayesian Search) hyperparameter optimization library with [Weights & Biases](https://wandb.ai/) (W&B) sweeps. It enables cost-aware Bayesian parameter search using W&B's sweep functionality, allowing for efficient hyperparameter optimization across multiple parallel agents while storing all state within W&B.

> **Note:** This project is using [uv](https://github.com/astral-sh/uv) for fast Python package management. Refer to the top-level README for instructions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [How It Works](#how-it-works)
  - [Creating a Sweep](#creating-a-sweep)
  - [Running an Experiment](#running-an-experiment)
- [Examples](#examples)
- [Notes](#notes)
- [References](#references)
- [License](#license)

## Features

- **Cost-Aware Bayesian Optimization**: Optimize hyperparameters considering both performance and computational cost.
- **Seamless W&B Integration**: Leverage W&B sweeps for distributed hyperparameter optimization with state stored in W&B.
- **Parallel Execution**: Support multiple parallel agents running as part of the sweep.
- **Automatic State Management**: Each agent initializes with observations from completed runs and generates suggestions for incomplete runs.
- **Customizable Parameter Spaces**: Support for `LogSpace`, `LogitSpace`, and `LinearSpace` parameter spaces.

## Installation

You can install Wandb Carbs using UV:

```bash
uv pip install wandb_carbs
```

For local development, refer to the top-level [README.md](../README.md) in this repository.

## Usage

### Prerequisites

- An account on [Weights & Biases](https://wandb.ai/).
- Basic understanding of [CARBS](https://github.com/imbue-ai/carbs) for Bayesian optimization.
- Familiarity with W&B sweeps.

### How It Works

CARBS performs cost-aware Bayesian parameter search but doesn't natively integrate with W&B sweeps. Wandb Carbs bridges this gap by:

- **Creating a W&B Sweep**: Converts CARBS parameter definitions into a W&B sweep configuration.
- **Initializing Agents**: Each sweep agent run creates a new CARBS instance.
- **State Initialization**: Agents initialize with observations from completed runs in the sweep and generate suggestions for incomplete runs.
- **Parameter Suggestion**: Generates a suggestion for the current run and updates the run's configuration with the suggested parameters.
- **Recording Results**: After the run completes, it stores the CARBS objective and cost in the run's metadata.

This setup allows multiple parallel agents to run as part of the sweep while storing all the state in W&B.

### Creating a Sweep

First, define your parameter spaces using CARBS:

```python
from carbs import Param, LogSpace, LinearSpace

# Define parameter spaces
params = [
    Param(name='learning_rate', space=LogSpace(min=1e-5, max=1e-1)),
    Param(name='batch_size', space=LinearSpace(min=16, max=128, is_integer=True)),
]
```

Use the `create_sweep` function from Wandb Carbs to create a W&B sweep:

```python
from wandb_carbs import create_sweep

sweep_id = create_sweep(
    sweep_name='My CARBS Sweep',
    wandb_entity='your_wandb_entity',
    wandb_project='your_wandb_project',
    carbs_spaces=params
)
```

This function converts the CARBS parameters into a W&B sweep configuration and returns a `sweep_id` that you can use to manage the sweep.

### Running an Experiment

In your training script, integrate Wandb Carbs:

```python
import wandb
from wandb_carbs import WandbCarbs
from carbs import CARBS

def train():
    # Initialize W&B run
    wandb.init()

    # Initialize CARBS
    carbs = CARBS(params=params)

    # Initialize Wandb Carbs
    wandb_carbs = WandbCarbs(carbs=carbs)

    # Get hyperparameters suggestion
    suggestion = wandb_carbs.suggest()

    # Your training code here
    # The suggested parameters are in wandb.config
    model = build_model(wandb.config)
    performance = evaluate_model(model)

    # Record observation
    objective = performance['accuracy']  # The metric you aim to optimize
    cost = performance['training_time']  # Computational cost measure
    wandb_carbs.record_observation(objective=objective, cost=cost)

    # Finish the run
    wandb.finish()

if __name__ == '__main__':
    train()
```

**Note:** The suggested hyperparameters are automatically stored in `wandb.config` by the W&B agent; therefore, you don't need to manually update `wandb.config` with the suggestion.

## Examples

### Full Example

```python
import wandb
from wandb_carbs import WandbCarbs, create_sweep
from carbs import CARBS, Param, LogSpace, LinearSpace

# Define parameter spaces
params = [
    Param(name='learning_rate', space=LogSpace(min=1e-5, max=1e-1)),
    Param(name='batch_size', space=LinearSpace(min=16, max=128, is_integer=True)),
]

# Create a sweep
sweep_id = create_sweep(
    sweep_name='My CARBS Sweep',
    wandb_entity='your_wandb_entity',
    wandb_project='your_wandb_project',
    carbs_spaces=params
)

def train():
    # Initialize W&B run
    wandb.init()

    # Initialize CARBS
    carbs = CARBS(params=params)

    # Initialize Wandb Carbs
    wandb_carbs = WandbCarbs(carbs=carbs)

    # Get hyperparameters suggestion
    suggestion = wandb_carbs.suggest()

    # Your training code here
    # The suggested parameters are in wandb.config
    model = build_model(wandb.config)
    performance = evaluate_model(model)

    # Record observation
    objective = performance['accuracy']
    cost = performance['training_time']
    wandb_carbs.record_observation(objective=objective, cost=cost)

    # Finish the run
    wandb.finish()

if __name__ == '__main__':
    train()
```

### Managing the Sweep

- Use the `sweep_id` returned by `create_sweep` to monitor and manage your sweep in the W&B dashboard.
- Ensure that all agents (parallel runs) are correctly configured to run the `train` function.

## Notes

- Replace `your_wandb_entity` and `your_wandb_project` with your actual W&B entity and project names.
- The `objective` should be the metric you aim to optimize (e.g., accuracy, F1 score).
- The `cost` can be any measure of computational cost, like training time, memory usage, or FLOPs.
- Ensure that your `build_model` and `evaluate_model` functions use the parameters from `wandb.config`.

## References

- **CARBS**: For more detailed instructions on CARBS, visit the [CARBS GitHub repository](https://github.com/imbue-ai/carbs).
- **Weights & Biases**: Learn more about W&B sweeps [here](https://docs.wandb.ai/guides/sweeps).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
