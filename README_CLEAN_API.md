# Metta Clean API

A clean, simple API for using Metta without Hydra configuration files.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
# Train a policy (default settings)
python train.py

# Train with custom settings
python train.py --timesteps 1000000 --batch-size 512 --num-agents 4

# Evaluate a checkpoint
python train.py --mode eval --checkpoint ./train_dir/default_run/checkpoints/policy_final.pt
```

## Architecture

The clean API consists of three main components:

1. **`train.py`** - Main entry point for training and evaluation
2. **`metta_api.py`** - Clean API module providing factory functions and utilities
3. **`examples/run_functional_training.py`** - Example of using the functional training API directly

## API Overview

### Factory Functions

The `metta_api` module provides these key functions:

- `make_agent()` - Create a Metta agent directly
- `make_vecenv()` - Create a vectorized environment
- `make_optimizer()` - Create an optimizer
- `make_loss_module()` - Create a loss module
- `make_experience_buffer()` - Create an experience buffer

### Configuration Functions

- `agent()` - Create agent configuration
- `env()` - Create environment configuration
- `trainer()` - Create trainer configuration
- `optimizer()` - Create optimizer configuration

### Utility Functions

- `build_runtime_config()` - Build runtime configuration
- `setup_metta_environment()` - Setup logging and environment
- `get_logger()` - Get a configured logger
- `quick_train()` - High-level training function
- `quick_eval()` - High-level evaluation function

## Example Usage

```python
import metta_api as metta

# Simple training
checkpoint = metta.quick_train(
    run_name="my_experiment",
    timesteps=100_000,
    batch_size=256,
    num_agents=2,
)

# Simple evaluation
results = metta.quick_eval(
    checkpoint_path=checkpoint,
    num_episodes=10,
)
print(f"Average reward: {results['avg_reward']:.4f}")
```

## Advanced Usage

For more control, use the functional training components directly:

```python
from metta.rl.functional_trainer import (
    perform_rollout_step,
    compute_initial_advantages,
    process_rollout_infos,
)

# See examples/run_functional_training.py for a complete example
```

## Migration from Hydra

The clean API eliminates the need for Hydra configuration files. Instead of:

```yaml
# config.yaml
trainer:
  batch_size: 256
  learning_rate: 3e-4
```

You can now use:

```python
config = metta.trainer(batch_size=256, learning_rate=3e-4)
```

All configuration is done programmatically in Python, making it easier to debug and integrate with other systems.
