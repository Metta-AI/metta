# Metta API Documentation

This document describes the clean API for using Metta as a library without Hydra configuration.

## Overview

The Metta API provides direct instantiation of training components without requiring Hydra configuration files. This allows you to:

1. Create environments, agents, and training components programmatically
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
4. Easily customize any part of the training process

## Quick Start

```python
#!/usr/bin/env -S uv run
import torch
from metta.api import (
    Environment, Agent, Optimizer,
    setup_run_directories, save_experiment_config,
    perform_rollout_step, process_minibatch_update,
    accumulate_rollout_stats, compute_advantage,
    calculate_anneal_beta
)
from metta.agent.policy_store import PolicyStore
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import TrainerConfig
from metta.common.profiling.stopwatch import Stopwatch
from omegaconf import DictConfig

# Set up directories
dirs = setup_run_directories()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and agent
env = Environment(num_agents=4, width=32, height=32, device=str(device))
agent = Agent(env, device=str(device))

# Create trainer config
trainer_config = TrainerConfig(
    total_timesteps=10_000_000,
    batch_size=16384,
    minibatch_size=512,
)

# Save experiment config
save_experiment_config(dirs, device, trainer_config)

# Create policy store
policy_store = PolicyStore(
    DictConfig({
        "device": str(device),
        "policy_cache_size": 10,
        "trainer": {
            "checkpoint": {
                "checkpoint_dir": dirs.checkpoint_dir,
            }
        },
    }),
    wandb_run=None,
)

# Create optimizer
optimizer = Optimizer(
    optimizer_type="adam",
    policy=agent,
    learning_rate=3e-4,
)

# Training loop
while agent_step < trainer_config.total_timesteps:
    # Rollout phase
    while not experience.ready_for_training:
        num_steps, info = perform_rollout_step(agent, env, experience, device, timer)
        agent_step += num_steps

    # Training phase
    for epoch in range(trainer_config.update_epochs):
        for minibatch in experience.iter_minibatches():
            loss = process_minibatch_update(...)
            optimizer.step(loss, epoch)
```

## Training Flow in run.py

The `run.py` example demonstrates a complete training flow using the Metta API:

### 1. Setup Phase
- **Directory Structure**: Create run directories for checkpoints, replays, and stats
- **Configuration**: Create TrainerConfig with all hyperparameters
- **Save Config**: Save the full experiment configuration for reproducibility

### 2. Component Creation
- **Environment**: Create a bucketed navigation curriculum environment with multiple terrain types
- **Agent**: Create a CNN-LSTM agent using default architecture
- **PolicyStore**: Direct instantiation with minimal config for checkpoint management
- **Optimizer**: Adam or Muon optimizer wrapper with gradient clipping
- **Experience Buffer**: Segmented storage for BPTT with configurable horizon
- **Kickstarter**: For distillation from teacher policies
- **Monitoring**: Memory monitor and system monitor for resource tracking

### 3. Training Loop
The main training loop follows this pattern:

#### Rollout Phase:
- Reset experience buffer for new rollout
- Collect environment steps until buffer is full
- Accumulate rollout statistics

#### Training Phase:
- Calculate advantages using GAE with V-trace corrections
- Train for multiple epochs on collected data
- Sample prioritized minibatches
- Process each minibatch with PPO loss
- Handle gradient accumulation and clipping
- Early exit if KL divergence exceeds target

#### Periodic Tasks:
- **Checkpointing**: Save policy and training state every N epochs
- **Evaluation**: Run policy on evaluation environments
- **Replay Generation**: Create visual replays for inspection
- **Monitoring**: Log system stats, gradient stats, and training metrics
- **L2 Weight Updates**: Update L2 regularization reference weights

### 4. Evaluation

The evaluation configuration uses different terrain sizes:
```python
evaluation_config = SimulationSuiteConfig(
    name="evaluation",
    simulations={
        "navigation/terrain_small": SingleEnvSimulationConfig(...),
        "navigation/terrain_medium": SingleEnvSimulationConfig(...),
        "navigation/terrain_large": SingleEnvSimulationConfig(...),
    },
)
```

### 5. Cleanup
- Save final checkpoint if needed
- Log evaluation history and final stats
- Stop monitors and close environment

## Key Components

### Environment

Create MettaGrid environments with or without curriculum:

```python
# Simple environment
env = Environment(
    num_agents=4,
    width=32,
    height=32,
    device="cuda",
)

# With navigation curriculum
env = Environment(
    curriculum_path="/env/mettagrid/curriculum/navigation/bucketed",
    num_agents=4,
    device="cuda",
)
```

### Agent

Create agents with default or custom configurations:

```python
# Default agent
agent = Agent(env, device="cuda")

# Custom agent config
config = DictConfig({
    "device": "cuda",
    "agent": {
        "clip_range": 0.2,
        "l2_init_weight_update_interval": 100,
        # ... component definitions ...
    }
})
agent = Agent(env, config=config)
```

### Optimizer

Wrapper for PyTorch optimizers with gradient accumulation:

```python
optimizer = Optimizer(
    optimizer_type="adam",  # or "muon"
    policy=agent,
    learning_rate=3e-4,
    max_grad_norm=0.5,
)

# Training step
loss = compute_loss(...)
optimizer.step(loss, epoch, accumulate_steps)
```

### Experience Buffer

Segmented tensor storage for BPTT:

```python
experience = Experience(
    total_agents=env.num_agents,
    batch_size=16384,
    bptt_horizon=16,
    minibatch_size=512,
    obs_space=env.single_observation_space,
    atn_space=env.single_action_space,
    device=device,
    hidden_size=512,
    num_lstm_layers=1,
)
```

### Policy Store

Manages policy checkpoints:

```python
# Create policy record
policy_record = policy_store.create_empty_policy_record(name)
policy_record.metadata = {"epoch": epoch, "score": score}
policy_record.policy = agent

# Save
saved_record = policy_store.save(policy_record)

# Load
loaded_record = policy_store.policy_record("path/to/checkpoint")
```

## Configuration Saving

Save experiment configuration using the provided helper:

```python
from metta.api import save_experiment_config, setup_run_directories
from metta.rl.trainer_config import TrainerConfig

# Set up directories
dirs = setup_run_directories()

# Create trainer config
trainer_config = TrainerConfig(
    total_timesteps=10_000_000,
    batch_size=16384,
    # ... other config ...
)

# Save configuration
save_experiment_config(dirs, device, trainer_config)
# Saves to {dirs.run_dir}/config.yaml with full experiment metadata
```

### Policy Store

Create a policy store directly with minimal configuration:

```python
from metta.agent.policy_store import PolicyStore
from omegaconf import DictConfig

# Create minimal config for PolicyStore
policy_store = PolicyStore(
    DictConfig({
        "device": "cuda",
        "policy_cache_size": 10,
        "trainer": {
            "checkpoint": {
                "checkpoint_dir": "./checkpoints",
            }
        },
    }),
    wandb_run=None,
)
```

### Checkpointing

Save policies using PolicyStore and PolicyRecord directly:

```python
# Create a policy record
name = policy_store.make_model_name(epoch)
policy_record = policy_store.create_empty_policy_record(name)
policy_record.metadata = {
    "agent_step": agent_step,
    "epoch": epoch,
    "score": 0.95,
    # any other metadata you want
}
policy_record.policy = agent

# Save through policy store
saved_record = policy_store.save(policy_record)

# Load checkpoint
loaded_record = policy_store.policy_record("path/to/checkpoint")
```

### compute_advantage

Compute advantages using GAE. Note that you need to create the initial tensors:

```python
from metta.api import compute_advantage

# Create initial tensors
advantages = torch.zeros(experience.values.shape, device=device)
initial_importance_sampling_ratio = torch.ones_like(experience.values)

# Compute advantages using GAE
advantages = compute_advantage(
    experience.values,
    experience.rewards,
    experience.dones,
    initial_importance_sampling_ratio,
    advantages,
    trainer_config.ppo.gamma,
    trainer_config.ppo.gae_lambda,
    trainer_config.vtrace.vtrace_rho_clip,
    trainer_config.vtrace.vtrace_c_clip,
    device,
)
```

## Design Philosophy

The API design follows these principles:

1. **Transparency**: All components are visible and their interactions are explicit
2. **Modularity**: Each component can be used independently
3. **Compatibility**: Uses the same underlying components as the Hydra-based training
4. **Type Safety**: Leverages Pydantic models for configuration validation
5. **Flexibility**: Easy to customize any part of the training process

This makes Metta accessible for:
- Researchers who need custom training loops
- Integration into existing ML pipelines
- Educational purposes to understand RL training
- Debugging and experimentation
