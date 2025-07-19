# Metta API Documentation

The Metta API (`metta.api`) provides a clean way to use Metta's training components without Hydra configuration files.

## Quick Start

```python
#!/usr/bin/env -S uv run
import torch
from metta.api.agent import Agent, wrap_agent_distributed
from metta.api.environment import Environment
from metta.api.directories import setup_run_directories, setup_device_and_distributed
from metta.api.training import Optimizer
from metta.agent.policy_store import PolicyStore
from metta.rl.experience import Experience
from metta.rl.trainer_config import TrainerConfig

# Setup
dirs = setup_run_directories()
device = setup_device_and_distributed("cuda" if torch.cuda.is_available() else "cpu")[0]

# Create environment and agent
env = Environment(curriculum_path="/env/mettagrid/curriculum/navigation/bucketed")
agent = Agent(env, device=str(device))

# Training loop (see run.py for complete example)
```

## Core Components

### Environment

```python
from metta.api.environment import Environment

# Simple environment
env = Environment(num_agents=4, width=32, height=32)

# With curriculum
env = Environment(curriculum_path="/env/mettagrid/curriculum/navigation/bucketed")
```

### Agent

```python
from metta.api.agent import Agent

# Default CNN-LSTM agent
agent = Agent(env, device="cuda")

# Custom config
from omegaconf import DictConfig
config = DictConfig({"agent": {...}})
agent = Agent(env, config=config)
```

### Optimizer

```python
from metta.api.training import Optimizer

optimizer = Optimizer(
    optimizer_type="adam",  # or "muon"
    policy=agent,
    learning_rate=3e-4,
)

# Training step
optimizer.step(loss, epoch)
```

### Training Loop Functions

```python
from metta.rl.functions import (
    perform_rollout_step,
    accumulate_rollout_stats,
    compute_advantage,
    process_minibatch_update,
)

# Rollout
while not experience.ready_for_training:
    num_steps, info = perform_rollout_step(agent, env, experience, device, timer)

# Process stats
accumulate_rollout_stats(info, stats)

# Train
advantages = compute_advantage(...)
loss = process_minibatch_update(...)
```

### Utilities

```python
from metta.api.directories import (
    setup_run_directories,
    setup_device_and_distributed,
    save_experiment_config,
)

from metta.api.training import (
    initialize_wandb,
    cleanup_wandb,
    cleanup_distributed,
    load_checkpoint,
    save_checkpoint,
)

from metta.api.evaluation import (
    create_evaluation_config_suite,
    evaluate_policy_suite,
    generate_replay_simple,
)
```

## Distributed Training

```python
# Setup
device = setup_device_and_distributed("cuda")
is_master, world_size, rank = setup_distributed_vars()

# Wrap agent for distributed
agent = wrap_agent_distributed(agent, device)

# Run with torchrun
# torchrun --nproc_per_node=4 run.py
```

## Checkpointing

```python
# Save
saved_policy = save_checkpoint(
    epoch=epoch,
    agent_step=agent_step,
    agent=agent,
    optimizer=optimizer,
    policy_store=policy_store,
    checkpoint_path=checkpoint_dir,
    checkpoint_interval=10,
)

# Load
agent_step, epoch, policy_path = load_checkpoint(
    checkpoint_dir=checkpoint_dir,
    agent=agent,
    optimizer=optimizer,
    policy_store=policy_store,
)
```

## Configuration

Use Pydantic models for type-safe configuration:

```python
from metta.rl.trainer_config import TrainerConfig, PPOConfig

trainer_config = TrainerConfig(
    total_timesteps=10_000_000,
    batch_size=16384,
    ppo=PPOConfig(gamma=0.99, gae_lambda=0.95),
)
```

## Complete Training Example

See `run.py` for a complete example of training without Hydra configuration files.

## Key Exports

The `metta.api`
