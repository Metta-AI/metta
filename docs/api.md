# Metta API Documentation

The Metta API (`metta.api`) provides a clean way to use Metta's training components without Hydra configuration files.

## Quick Start

```python
#!/usr/bin/env -S uv run
import torch
from metta.api import (
    Agent, Environment, Optimizer,
    setup_device_and_distributed, setup_run_directories,
    save_checkpoint, load_checkpoint, wrap_agent_distributed,
)
from metta.agent.policy_store import PolicyStore
from metta.rl.experience import Experience
from metta.rl.trainer_config import TrainerConfig

# Setup
dirs = setup_run_directories()
device = setup_device_and_distributed("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and agent
env = Environment(curriculum_path="/env/mettagrid/curriculum/navigation/bucketed")
agent = Agent(env, device=str(device))

# Training loop (see run.py for complete example)
```

## Core Components

### Environment

```python
# Simple environment
env = Environment(num_agents=4, width=32, height=32)

# With curriculum
env = Environment(curriculum_path="/env/mettagrid/curriculum/navigation/bucketed")
```

### Agent

```python
# Default CNN-LSTM agent
agent = Agent(env, device="cuda")

# Custom config
from omegaconf import DictConfig
config = DictConfig({"agent": {...}})
agent = Agent(env, config=config)
```

### Optimizer

```python
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
from metta.api import (
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

## Complete Example

See `run.py` for a complete training implementation that includes:

- Environment creation with curriculum
- Agent initialization
- Distributed training support
- Checkpointing and recovery
- Evaluation and replay generation
- Monitoring and logging

## Key Exports

The `metta.api` module exports:

**Factories**: `Environment`, `Agent`, `Optimizer`

**Training**: `perform_rollout_step`, `process_minibatch_update`, `accumulate_rollout_stats`, `compute_advantage`

**Distributed**: `setup_device_and_distributed`, `setup_distributed_vars`, `wrap_agent_distributed`,
`cleanup_distributed`

**Checkpointing**: `save_checkpoint`, `load_checkpoint`

**Configuration**: `TrainerConfig`, `PPOConfig`, `OptimizerConfig`, `CheckpointConfig`

**Components**: `Experience`, `Kickstarter`, `Losses`, `Stopwatch`

**Utilities**: `setup_run_directories`, `save_experiment_config`, `create_evaluation_config_suite`,
`create_replay_config`
