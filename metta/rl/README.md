# Modular RL Components

This directory contains refactored, modular RL training components designed for flexibility and ease of customization.

## Overview

The refactoring splits the monolithic `MettaTrainer` into composable components:

- **RolloutCollector**: Handles environment interaction and data collection
- **PPOOptimizer**: Implements PPO algorithm with support for custom losses
- **PolicyEvaluator**: Evaluates policies on simulation suites
- **TrainingCheckpointer**: Manages saving/loading of training state
- **StatsLogger**: Handles metrics logging to wandb and console

## Key Features

### 1. Structured Configs

Replace YAML configs with typed dataclasses:

```python
from metta.train.train_config import TrainingConfig, small_fast_config

# Use preset
config = small_fast_config()

# Or create custom
config = TrainingConfig(
    experiment_name="my_experiment",
    agent=AgentConfig(name="simple_cnn", hidden_size=512),
    trainer=TrainerConfig(total_timesteps=1_000_000),
)
```

### 2. Functional Training Loop

Explicit, modifiable training loop:

```python
from metta.rl.functional_trainer import functional_training_loop

state = functional_training_loop(
    config=trainer_config,
    ppo_config=ppo_config,
    policy=agent,
    curriculum=curriculum,
    policy_store=policy_store,
    step_fn=custom_training_step,  # Optional custom step
    custom_losses=[curiosity_loss, diversity_loss],  # Optional custom losses
)
```

### 3. Custom Loss Functions

Easy to add custom losses:

```python
def curiosity_loss(policy, obs, actions, rewards, values, advantages):
    """Custom loss for exploration."""
    # Your implementation
    return loss_tensor

# Pass to PPO optimizer
ppo_optimizer.update(experience, custom_loss_fns=[curiosity_loss])
```

### 4. Direct Component Usage

Use components directly without the full trainer:

```python
# Create components
collector = RolloutCollector(vecenv, policy, experience, device)
ppo = PPOOptimizer(policy, optimizer, device, config)

# Training loop
for epoch in range(1000):
    # Collect data
    stats, steps = collector.collect()

    # Update policy
    losses = ppo.update(experience)

    # Your custom logic here
```

## Migration Guide

### From Legacy YAML Config

```python
# Old YAML config
yaml_config = {
    "agent": {"name": "simple_cnn"},
    "trainer": {"total_timesteps": 10_000_000},
}

# Convert to structured config
config = TrainingConfig.from_dict(yaml_config)
```

### From MettaTrainer to Modular Components

Old:
```python
trainer = MettaTrainer(cfg, wandb_run, policy_store, sim_config, stats_client)
trainer.train()
```

New:
```python
# Option 1: Use functional trainer with same interface
state = functional_training_loop(config, ppo_config, policy, curriculum, policy_store)

# Option 2: Use components directly for full control
components = create_training_components(config, ppo_config, policy, vecenv, experience, policy_store)
while state.agent_steps < config.total_timesteps:
    metrics = default_training_step(state, components, config, experience)
```

## Examples

See the `examples/` directory for:
- `direct_training_with_components.py` - Using components directly
- `functional_training_example.py` - Functional training with custom losses
- `structured_config_training.py` - Using structured configs

## Component Details

### PPOOptimizer

Core PPO implementation with clean separation of loss computation:

```python
ppo = PPOOptimizer(policy, optimizer, device, PPOConfig())

# Get individual loss components
losses = ppo.compute_ppo_loss(obs, actions, old_logprobs, advantages, returns, values)
# Returns: policy_loss, value_loss, entropy_loss, total_loss, approx_kl, clipfrac

# Or run full update
stats = ppo.update(experience, custom_loss_fns=[my_custom_loss])
```

### RolloutCollector

Handles environment stepping and data collection:

```python
collector = RolloutCollector(vecenv, policy, experience, device)
stats, num_steps = collector.collect()  # Fills experience buffer
```

### TrainingCheckpointer

Simple checkpoint management:

```python
checkpointer = TrainingCheckpointer(checkpoint_dir, policy_store, wandb_run)

# Save
policy_record = checkpointer.save_policy(policy, epoch, metadata)
checkpointer.save_trainer_state(agent_step, epoch, optimizer.state_dict(), run_dir)

# Load
checkpoint = checkpointer.load_trainer_state(run_dir)
policy_record = checkpointer.load_policy(checkpoint, env)
```

## Design Principles

1. **Separation of Concerns**: Each component has a single, clear responsibility
2. **Composability**: Components can be mixed and matched as needed
3. **Explicit Control**: Training loop is visible and modifiable
4. **Type Safety**: Structured configs with type hints
5. **Backward Compatibility**: Can still load from YAML configs
