# Metta Refactoring Guide

This document explains the major refactoring of the Metta library to make it more pythonic and easier to use.

## Overview of Changes

### 1. **Functional Training API**

We've broken down the monolithic `MettaTrainer` class into functional components in `metta/train/`:

```python
from metta.train import (
    rollout,              # Collect experience from environments
    compute_ppo_loss,     # Calculate PPO losses
    update_policy,        # Perform gradient updates
    save_checkpoint,      # Save training state
    evaluate_policy,      # Evaluate on test environments
)
```

This makes the training loop explicit and customizable:

```python
# Training loop
while training:
    # Collect experience
    stats = rollout(agent, vecenv, experience, rollout_config)

    # Compute losses
    loss, losses = compute_ppo_loss(agent, experience, ppo_config)

    # Update policy
    update_policy(agent, optimizer, loss, opt_config)
```

### 2. **Python-Based Configuration**

Instead of complex YAML files with Hydra, configurations are now Python code in `configs/python/`:

**Agent Configs** (`configs/python/agents.py`):
```python
from configs.python.agents import simple_cnn_agent, large_cnn_agent

# Get agent configuration
agent_config = simple_cnn_agent()
agent = MettaAgent(**agent_config)
```

**Environment Configs** (`configs/python/environments.py`):
```python
from configs.python.environments import NavigationWalls, MemorySequence

# Create environment
env_config = NavigationWalls(width=40, height=40)
env = create_env(env_config)
```

### 3. **Simplified Tools**

New, clearer entrypoints in `tools/`:

- **`train_new.py`** - Training with explicit object creation
- **`evaluate_new.py`** - Policy evaluation without YAML configs
- **`play_new.py`** - Interactive play or watch trained agents

Example usage:
```bash
# Train an agent
python tools/train_new.py --run-name my_experiment --total-timesteps 10000000

# Evaluate a checkpoint
python tools/evaluate_new.py checkpoints/latest.pt --env-suite navigation

# Play interactively
python tools/play_new.py --mode human --env navigation/maze

# Watch a trained agent
python tools/play_new.py --mode agent --agent checkpoints/latest.pt
```

### 4. **Direct Object Creation**

Instead of hiding object creation in YAML, we create objects directly:

```python
# Create agent
agent = MettaAgent(
    observations={"obs_key": "grid_obs"},
    components={...}  # Define architecture in Python
)

# Create optimizer
optimizer = torch.optim.Adam(
    agent.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999)
)

# Create experience buffer
experience = ExperienceBuffer(
    num_envs=128,
    rollout_length=128,
    device="cuda"
)
```

## Migration Guide

### From Old Training Script to New

**Old way (with Hydra)**:
```bash
python tools/train.py +env=mettagrid/navigation/simple trainer=puffer
```

**New way**:
```python
# In your training script
from configs.python.environments import NavigationWalls
from configs.python.agents import simple_cnn_agent

# Create environment
env_config = NavigationWalls()
env = create_env(env_config)

# Create agent
agent_config = simple_cnn_agent()
agent = MettaAgent(**agent_config)

# Train
while training:
    rollout(...)
    compute_ppo_loss(...)
    update_policy(...)
```

### From YAML Configs to Python

**Old YAML config**:
```yaml
game:
  map_builder:
    _target_: mettagrid.room.mean_distance.MeanDistance
    width: 35
    height: 35
    objects:
      altar: 3
      wall: 12
```

**New Python config**:
```python
class NavigationWalls(EnvConfig):
    def get_map_builder(self):
        return {
            "_target_": "mettagrid.room.mean_distance.MeanDistance",
            "width": self.width,
            "height": self.height,
            "objects": {"altar": 3, "wall": 12},
        }
```

## Benefits

1. **Transparency** - You can see exactly what objects are being created
2. **Flexibility** - Easy to modify training loops and add custom logic
3. **Debuggability** - Standard Python debugging tools work naturally
4. **Type Safety** - IDEs can provide better autocomplete and type checking
5. **Simplicity** - No need to understand Hydra's complex override system

## Next Steps

1. The old YAML configs and Hydra-based tools still exist for backward compatibility
2. Gradually migrate your experiments to use the new Python-based approach
3. Create custom training scripts that fit your specific needs
4. Define new environment and agent configurations as Python classes

## Example: Custom Training Script

Here's a complete example of a custom training script using the new API:

```python
#!/usr/bin/env python
"""My custom training script."""

import torch
from metta.train import rollout, compute_ppo_loss, update_policy, save_checkpoint
from metta.agent import MettaAgent
from metta.rl.experience import Experience
from configs.python.agents import simple_cnn_agent
from configs.python.environments import NavigationWalls

# Setup
device = "cuda"
env_config = NavigationWalls()
env = create_env(env_config)

agent_config = simple_cnn_agent()
agent = MettaAgent(**agent_config).to(device)

optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
experience = Experience(...)

# Training loop
for epoch in range(1000):
    # Collect data
    stats = rollout(agent, env, experience, rollout_config)

    # Update policy
    for _ in range(4):  # PPO epochs
        loss, losses = compute_ppo_loss(agent, experience, ppo_config)
        update_policy(agent, optimizer, loss, opt_config)

    # Log and save
    print(f"Epoch {epoch}: {losses.policy_loss:.4f}")
    if epoch % 100 == 0:
        save_checkpoint("checkpoints", agent, optimizer, epoch, agent_step)
```

This approach gives you full control over the training process while still benefiting from Metta's optimized components.
