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
from omegaconf import DictConfig, OmegaConf

# Set up directories
dirs = setup_run_directories()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and agent
env = Environment(num_agents=4, width=32, height=32, device=str(device))
agent = Agent(env, device=str(device))

# Create training components individually
optimizer = Optimizer(optimizer_type="adam", policy=agent)
experience = Experience(...)  # Configure as needed
kickstarter = Kickstarter(...)  # Configure as needed
losses = Losses()
timer = Stopwatch()

# Training loop with full control
agent_step = 0
epoch = 0
stats = {}

while agent_step < total_timesteps:
    # Rollout phase
    experience.reset_for_rollout()
    while not experience.ready_for_training:
        num_steps, info = perform_rollout_step(agent, env, experience, device, timer)
        agent_step += num_steps

    accumulate_rollout_stats(info, stats)

    # Training phase
    losses.zero()
    experience.reset_importance_sampling_ratios()

    # Compute advantages directly
    advantages = torch.zeros(experience.values.shape, device=device)
    initial_importance_sampling_ratio = torch.ones_like(experience.values)

    advantages = compute_advantage(
        experience.values, experience.rewards, experience.dones,
        initial_importance_sampling_ratio, advantages,
        ppo_config.gamma, ppo_config.gae_lambda,
        vtrace_config.vtrace_rho_clip, vtrace_config.vtrace_c_clip,
        device,
    )

    # Train on minibatches
    for minibatch in experience.sample_minibatch(...):
        loss = process_minibatch_update(...)
        optimizer.step(loss, epoch, accumulate_steps)

    epoch += 1
```

## Core Components

### Environment

Create MettaGrid environments without Hydra:

```python
from metta.api import Environment

# Simple environment with convenience parameters
env = Environment(
    num_agents=4,
    width=32,
    height=32,
    device="cuda",
    num_envs=64,
    num_workers=4,
)

# Advanced: Use a specific curriculum
env = Environment(
    curriculum_path="/env/mettagrid/navigation",
    device="cuda",
    num_envs=64,
)

# Advanced: Provide custom environment config
env_config = {
    "game": {
        "max_steps": 1000,
        "num_agents": 4,
        # ... full config
    }
}
env = Environment(env_config=env_config)
```

### Agent

Create Metta agents programmatically:

```python
from metta.api import Agent

# Create agent with default configuration
agent = Agent(env, device="cuda")

# Advanced: Custom agent configuration
from omegaconf import DictConfig

agent_config = DictConfig({
    "device": "cuda",
    "agent": {
        "clip_range": 0,
        "analyze_weights_interval": 300,
        # ... full agent config
    }
})
agent = Agent(env, config=agent_config)
```

### Optimizer

The Optimizer wrapper provides a clean interface for optimization with gradient accumulation:

```python
from metta.api import Optimizer

# Create optimizer with default settings
optimizer = Optimizer(
    optimizer_type="adam",  # or "muon"
    policy=agent,
    learning_rate=3e-4,
    max_grad_norm=0.5,
)

# Training step with gradient accumulation
optimizer.step(loss, epoch, accumulate_steps=4)

# Access underlying PyTorch optimizer if needed
learning_rate = optimizer.param_groups[0]["lr"]
```

### Experience Buffer

The Experience buffer stores rollout data with BPTT support:

```python
from metta.rl.experience import Experience

experience = Experience(
    total_agents=env.num_agents,
    batch_size=16384,
    bptt_horizon=16,
    minibatch_size=512,
    max_minibatch_size=512,
    obs_space=env.single_observation_space,
    atn_space=env.single_action_space,
    device=device,
    hidden_size=512,  # LSTM hidden size
    num_lstm_layers=1,
)

# Reset before rollout
experience.reset_for_rollout()

# Check if ready for training
if experience.ready_for_training:
    # Sample minibatches
    for i in range(experience.num_minibatches):
        minibatch = experience.sample_minibatch(...)
```

### Training Configuration

Use Pydantic models for type-safe configuration:

```python
from metta.rl.trainer_config import (
    TrainerConfig,
    PPOConfig,
    OptimizerConfig,
    CheckpointConfig,
)

# Create configuration with IDE support and validation
trainer_config = TrainerConfig(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=16384,
    minibatch_size=512,
    ppo=PPOConfig(
        clip_coef=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    ),
    optimizer=OptimizerConfig(
        type="adam",
        learning_rate=3e-4,
        betas=(0.9, 0.999),
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir="./checkpoints",
        checkpoint_interval=100,
    ),
)
```

## Training Functions

The API provides direct access to the core training functions:

### perform_rollout_step

Perform a single environment step and store experience:

```python
from metta.api import perform_rollout_step

num_steps, info = perform_rollout_step(
    policy=agent,
    vecenv=env,
    experience=experience,
    device=device,
    timer=timer,  # Optional
)
```

### process_minibatch_update

Process a training update on a minibatch:

```python
from metta.api import process_minibatch_update

loss = process_minibatch_update(
    policy=agent,
    experience=experience,
    minibatch=minibatch,
    advantages=advantages,
    trainer_cfg=trainer_config,
    kickstarter=kickstarter,
    agent_step=agent_step,
    losses=losses,
    device=device,
)
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

### accumulate_rollout_stats

Accumulate statistics from rollout info dictionaries:

```python
from metta.api import accumulate_rollout_stats

stats = {}
accumulate_rollout_stats(raw_infos, stats)
```

## Helper Utilities

### Run Directory Setup

Set up the standard directory structure:

```python
from metta.api import setup_run_directories

dirs = setup_run_directories(
    run_name="my_experiment",  # Optional
    data_dir="./experiments",  # Optional
)

print(dirs.run_dir)         # Main run directory
print(dirs.checkpoint_dir)  # Checkpoint directory
print(dirs.replay_dir)      # Replay directory
print(dirs.stats_dir)       # Stats directory
print(dirs.run_name)        # Run name
```

### Configuration Saving

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

## Advanced Features

### Evaluation

Run policy evaluation:

```python
from metta.sim.simulation_suite import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

eval_config = SimulationSuiteConfig(
    name="evaluation",
    simulations={
        "navigation/simple": SingleEnvSimulationConfig(
            env="/env/mettagrid/simple",
            num_episodes=5,
            max_time_s=30,
        ),
    },
)

sim_suite = SimulationSuite(
    config=eval_config,
    policy_pr=policy_record,
    policy_store=policy_store,
    device=device,
    vectorization="serial",
    stats_dir="./stats",
)

results = sim_suite.simulate()
```

### Replay Generation

Generate replays for visualization:

```python
from metta.sim.simulation import Simulation

replay_config = SingleEnvSimulationConfig(
    env="/env/mettagrid/simple",
    num_episodes=1,
    max_time_s=60,
)

replay_sim = Simulation(
    name=f"replay_epoch_{epoch}",
    config=replay_config,
    policy_pr=policy_record,
    policy_store=policy_store,
    device=device,
    replay_dir="./replays",
)

results = replay_sim.simulate()
replay_urls = results.stats_db.get_replay_urls()
```

### System Monitoring

Monitor system resources during training:

```python
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.util.system_monitor import SystemMonitor

# Memory monitoring
memory_monitor = MemoryMonitor()
memory_monitor.add(experience, name="Experience", track_attributes=True)
memory_monitor.add(agent, name="Agent")

# System monitoring
system_monitor = SystemMonitor(
    sampling_interval_sec=1.0,
    history_size=100,
    auto_start=True,
)

# Get stats
system_stats = system_monitor.get_summary()
memory_stats = memory_monitor.stats()
```

### Gradient Statistics

Track gradient behavior:

```python
from metta.rl.functions import compute_gradient_stats

grad_stats = compute_gradient_stats(agent)
logger.info(
    f"Gradient stats - mean: {grad_stats['grad/mean']:.2e}, "
    f"variance: {grad_stats['grad/variance']:.2e}, "
    f"norm: {grad_stats['grad/norm']:.2e}"
)
```

## Checkpointing

Save policies using PolicyStore and PolicyRecord directly:

```python
from metta.api import PolicyRecord

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

## Complete Example

See `run.py` for a complete example that demonstrates:

- Setting up all components
- Implementing the full training loop
- Checkpointing and recovery
- Evaluation and replay generation
- System monitoring
- All features working together

## Key Differences from Hydra-based Training

- **Explicit component creation**: You create each component directly rather than through Hydra instantiation
- **Direct control**: The training loop is visible and customizable
- **No YAML files**: All configuration is done in Python with type safety
- **Modular design**: Each component can be used independently

This approach is ideal for:
- Research requiring custom training loops
- Integration into existing codebases
- Understanding how Metta works internally
- Experimentation with individual components

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
