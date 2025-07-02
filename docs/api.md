# Metta API Documentation

The Metta API provides a clean, programmatic interface to the Metta training framework without requiring Hydra configuration files. This allows you to use Metta components directly in your Python code while maintaining full compatibility with the existing infrastructure.

## Quick Start

First, ensure the package is installed:
```bash
./setup_api.sh
```

Then run the example:
```bash
python run.py
```

## Design Philosophy

The API provides:
- **Direct instantiation** of environments, agents, and training components
- **Pydantic configuration classes** for type-safe, validated settings
- **Exposed training loop components** giving you full visibility and control
- **Real training infrastructure** - not simplified wrappers, but the actual components used by MettaTrainer

## Overview

## Configuration Classes

Metta uses Pydantic models for configuration, providing validation and type safety:

### TrainerConfig
The main configuration class containing all training parameters:

```python
from metta.api import TrainerConfig, create_default_trainer_config

# Option 1: Use helper with defaults
config = create_default_trainer_config(
    num_workers=4,
    total_timesteps=50_000_000,
    batch_size=524288,
    ppo={"clip_coef": 0.2, "ent_coef": 0.01}
)

# Option 2: Create directly
config = TrainerConfig(
    num_workers=4,
    total_timesteps=50_000_000,
    batch_size=524288,
    minibatch_size=16384,
    ppo=PPOConfig(clip_coef=0.2),
    optimizer=OptimizerConfig(learning_rate=3e-4),
)
```

### PPOConfig
PPO algorithm hyperparameters:

```python
from metta.api import PPOConfig

ppo_config = PPOConfig(
    clip_coef=0.1,        # Policy clipping coefficient
    ent_coef=0.01,        # Entropy bonus coefficient
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda
    vf_coef=0.5,          # Value function loss coefficient
    max_grad_norm=0.5,    # Gradient clipping
)
```

### OptimizerConfig
Optimizer settings:

```python
from metta.api import OptimizerConfig

optimizer_config = OptimizerConfig(
    type="adam",          # "adam" or "muon"
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
)
```

## Core Components

### Environment
Factory for creating vectorized MettaGrid environments:

```python
env = Environment(
    curriculum_path="/env/mettagrid/simple",
    num_envs=64,
    num_workers=4,
    batch_size=16,
    device="cuda",
    zero_copy=True,
)
```

### Agent
Factory for creating Metta agents:

```python
from omegaconf import DictConfig

agent_config = DictConfig({
    "device": "cuda",
    "agent": {
        "hidden_dim": 512,
        "lstm_layers": 1,
        "bptt_horizon": 8,
    }
})

agent = Agent(env, agent_config, device="cuda")
```

### TrainingComponents
Container exposing all internal training components:

```python
training = TrainingComponents.create(
    vecenv=env,
    policy=agent,
    trainer_config=trainer_config,
    policy_store=policy_store,  # Optional, for checkpointing
)

# Access to all components:
# training.experience - Experience buffer
# training.optimizer - Torch optimizer
# training.losses - Loss tracking
# training.kickstarter - Kickstarter for teacher distillation
# training.timer - Performance timing
```

## Training Loop Control

The API exposes the actual training loop components, giving you full control:

### Rollout Phase
```python
# Reset for new rollout
training.reset_for_rollout()

# Collect experience
raw_infos = []
while not training.is_ready_for_training():
    num_steps, info = training.rollout_step()
    training.agent_step += num_steps
    if info:
        raw_infos.extend(info)

# Process statistics
training.accumulate_stats(raw_infos)
```

### Training Phase
```python
# Reset training state
training.reset_training_state()

# Compute advantages
advantages = training.compute_advantages()

# Calculate annealed beta for prioritized replay
anneal_beta = calculate_anneal_beta(
    epoch=training.epoch,
    total_timesteps=trainer_config.total_timesteps,
    batch_size=trainer_config.batch_size,
    prio_alpha=prio_cfg.prio_alpha,
    prio_beta0=prio_cfg.prio_beta0,
)

# Train on minibatches
for epoch in range(trainer_config.update_epochs):
    for i in range(training.experience.num_minibatches):
        # Sample minibatch with prioritized replay
        minibatch = training.sample_minibatch(
            advantages, i, total_minibatches, anneal_beta
        )

        # Compute loss and update
        loss = training.train_minibatch(minibatch, advantages)
        training.optimize_step(loss, accumulate_steps)
```

## Checkpointing

Save and load policies using PolicyStore:

```python
from metta.agent.policy_store import PolicyStore
from metta.api import save_checkpoint, load_checkpoint

# Create policy store
policy_store = PolicyStore(checkpoint_dir="./checkpoints")

# Save checkpoint
policy_record = save_checkpoint(
    policy=agent,
    policy_store=policy_store,
    epoch=100,
    metadata={"score": 0.95}
)

# Load checkpoint
loaded_record = load_checkpoint(policy_store, "path/to/checkpoint")
```

## Object Type Constants

The API exports constants for MettaGrid object types:

```python
from metta.api import (
    TYPE_AGENT,
    TYPE_WALL,
    TYPE_MINE_RED,
    TYPE_GENERATOR_RED,
    TYPE_ALTAR,
    # ... etc
)
```

## Complete Training Example

See `run.py` for a complete example that demonstrates:
- Creating environments and agents
- Setting up training components
- Running the full training loop
- Checkpointing
- Logging and monitoring

## Key Differences from Hydra-based Training

1. **Direct instantiation** - No Hydra configuration files needed
2. **Pydantic configs** - Type-safe configuration with validation
3. **Exposed components** - Direct access to Experience, Losses, etc.
4. **Manual control** - You control when rollouts happen, when to train, when to save
5. **Same infrastructure** - Uses the exact same components as MettaTrainer internally

## Implementation Notes

The API is designed to be a thin wrapper around Metta's core functionality:
- `Environment` wraps `make_vecenv` and curriculum creation
- `Agent` wraps `make_policy` with proper initialization
- `TrainingComponents` exposes the internal components that `MettaTrainer` uses
- All training functions (`perform_rollout_step`, `process_minibatch_update`, etc.) are the actual implementations from `metta.rl.functions`

This ensures that training behavior is identical whether you use the API or the traditional Hydra-based approach.

## Troubleshooting

### ImportError: No module named 'metta.mettagrid.mettagrid_c'

If you get this error, the package needs to be installed:
```bash
uv sync --inexact
```

Or if you don't have uv:
```bash
pip install -e .
```

### "Unable to cast Python instance to C++ type" Error

This error typically means the C++ extension needs to be rebuilt. This can happen after:
- Pulling changes from git
- Switching branches
- Updating dependencies

To fix, reinstall the package:
```bash
uv sync --inexact --reinstall-package metta-mettagrid
```

## Advanced Usage
