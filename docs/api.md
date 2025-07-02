# Metta API Documentation

The Metta API provides a simple interface for using Metta as a library without complex Hydra configuration management. This allows you to integrate Metta into your own training pipelines and experiments.

## Overview

The API provides:
- **Typed configuration classes** for structured, validated settings
- **Constructor-style factory classes** for creating agents, environments, optimizers, etc.
- **Training and evaluation functions** for standard RL workflows
- **Full control over the training loop** - no hidden logic or wrappers

## Typed Configuration Classes

Metta uses typed configuration classes to provide structure and validation:

### AgentModelConfig
```python
from metta.api import AgentModelConfig

config = AgentModelConfig(
    hidden_dim=1024,
    lstm_layers=1,
    use_prev_action=True,
    use_prev_reward=True,
    mlp_layers=2,
    bptt_horizon=8,
    forward_lstm=True,
    backbone="cnn",
    obs_scale=1,
    clip_range=0
)
```

### EnvConfig
```python
from metta.api import EnvConfig

config = EnvConfig(
    game={
        "max_steps": 1000,
        "num_agents": 64,
        "width": 64,
        "height": 64,
        "observation_height": 11,
        "observation_width": 11
    }
)
```

### OptimizerConfig
```python
from metta.api import OptimizerConfig

config = OptimizerConfig(
    type="adam",  # or "muon"
    learning_rate=0.0004,
    beta1=0.9,
    beta2=0.999,
    eps=1e-12,
    weight_decay=0
)
```

### PPOConfig
```python
from metta.api import PPOConfig

config = PPOConfig(
    clip_coef=0.1,
    ent_coef=0.0021,
    gae_lambda=0.916,
    gamma=0.977,
    max_grad_norm=0.5,
    vf_clip_coef=0.1,
    vf_coef=0.44,
    norm_adv=True,
    clip_vloss=True
)
```

### ExperienceConfig
```python
from metta.api import ExperienceConfig

config = ExperienceConfig(
    batch_size=524288,
    minibatch_size=16384,
    bptt_horizon=64,
    update_epochs=1,
    zero_copy=True,
    cpu_offload=False,
    async_factor=2
)
```

### CheckpointConfig
```python
from metta.api import CheckpointConfig

config = CheckpointConfig(
    checkpoint_interval=60,
    wandb_checkpoint_interval=300,
    checkpoint_dir="./checkpoints"
)
```

### SimulationConfig
```python
from metta.api import SimulationConfig

config = SimulationConfig(
    evaluate_interval=300,
    replay_interval=300,
    replay_dir="./replays"
)
```

## Core Factory Classes

### Creating Components

#### Environment
```python
# Simple usage
env = Environment()  # Default config

# With parameters
env = Environment(num_agents=4, width=32, height=32)

# With config object
env = Environment(config=EnvConfig(...))

# Mix config and parameters
env = Environment(
    config=EnvConfig(...),
    num_agents=8  # Override config value
)
```

#### Agent
```python
# Create agent with default config
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=torch.device("cuda")
)

# With parameters
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    hidden_dim=512,
    lstm_layers=2
)

# With config object
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=AgentModelConfig(hidden_dim=1024)
)
```

#### Optimizer
```python
# Default Adam optimizer
optimizer = Optimizer(agent)

# With parameters
optimizer = Optimizer(agent, learning_rate=1e-4, weight_decay=0.01)

# With config
optimizer = Optimizer(agent, config=OptimizerConfig(type="adam", learning_rate=3e-4))
```

#### ExperienceManager
```python
# Default config
experience = ExperienceManager(env, agent)

# With parameters
experience = ExperienceManager(env, agent, batch_size=8192, minibatch_size=512)

# With config
experience = ExperienceManager(env, agent, config=ExperienceConfig(...))
```

### Training Functions

#### rollout
```python
# Collect experience
batch_info = rollout(experience, agent, num_steps=None)
```

#### compute_advantages
```python
# Compute advantages using GAE
advantages = compute_advantages(
    experience,
    gamma=0.977,
    gae_lambda=0.916
)
```

#### train_ppo
```python
# Train using PPO
stats = train_ppo(
    agent,
    optimizer,
    experience,
    ppo_config=PPOConfig(...),  # Optional, can use kwargs
    update_epochs=4
)
```

### Utility Functions

#### save_checkpoint / load_checkpoint
```python
# Save agent state
save_checkpoint(agent, "./checkpoints", epoch=100)

# Load agent state
metadata = load_checkpoint(agent, "./checkpoints/checkpoint_000100.pt")
```

#### eval_policy
```python
# Evaluate a policy
results = eval_policy(
    agent,
    env,
    num_episodes=10
)
```

## Example Usage

### Basic Training Loop
```python
import torch
from metta.api import *

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and agent
env = Environment(num_agents=4)
agent = Agent(
    env.single_observation_space,
    env.single_action_space,
    [],
    device,
    hidden_dim=512
)

# Create optimizer and experience manager
optimizer = Optimizer(agent, learning_rate=3e-4)
experience = ExperienceManager(env, agent)

# Training loop
for epoch in range(100):
    # Collect experience
    batch_info = rollout(experience, agent)

    # Compute advantages
    advantages = compute_advantages(experience)

    # Train
    stats = train_ppo(agent, optimizer, experience, update_epochs=4)

    print(f"Epoch {epoch}: {stats}")

    # Save periodically
    if epoch % 10 == 0:
        save_checkpoint(agent, "./checkpoints", epoch)
```

### Using Typed Configs
```python
# Define all configs upfront
env_config = EnvConfig(
    game={"max_steps": 1000, "num_agents": 8}
)

agent_config = AgentModelConfig(
    hidden_dim=1024,
    lstm_layers=2,
    bptt_horizon=16
)

ppo_config = PPOConfig(
    clip_coef=0.2,
    ent_coef=0.01,
    gamma=0.99
)

# Create components with configs
env = Environment(config=env_config)
agent = Agent(..., config=agent_config)

# Train with config
stats = train_ppo(..., ppo_config=ppo_config)
```

### Mixing Configs and Parameters
```python
# You can mix typed configs with parameter overrides
env = Environment(
    config=EnvConfig(),  # Use defaults
    num_agents=16,      # Override specific values
    max_steps=2000
)

# Or skip configs entirely and use parameters
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    hidden_dim=256,
    lstm_layers=1
)
```

## Design Philosophy

The API is designed to:

1. **Provide direct access** - No hidden configuration management or complex wrappers
2. **Use typed configs** - Optional structured configuration with validation
3. **Allow flexibility** - Mix configs with parameters, or use parameters only
4. **Give full control** - You write the training loop, we provide the building blocks
5. **Be framework-agnostic** - Integrate into any training pipeline

## Working with Curriculums

For advanced use cases, you can create curriculums directly:

```python
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from omegaconf import DictConfig

# Create a single task curriculum
curriculum = SingleTaskCurriculum("/env/mettagrid/simple", DictConfig({}))

# Use it to create an environment
env = MettaGridEnv(curriculum=curriculum, render_mode=None)
```

```python
from metta.api import EnvConfig

# Define your environment
env_config = EnvConfig(
    game={
        "num_agents": 2,
        "width": 15,
        "height": 10,
        "max_steps": 1000,
    }
)
