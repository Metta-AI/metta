# Metta Refactoring Guide: From Hydra to Library Architecture

## Overview

This guide documents the comprehensive refactoring of the Metta codebase from a Hydra-based configuration system to a modular Python library architecture. The goal is to make the codebase more pythonic, easier to understand, and simpler to use as a library.

## Key Improvements

### 1. Direct Object Creation

**Before (YAML + Hydra):**
```yaml
# Multiple YAML files with complex inheritance
defaults:
  - common
  - agent: simple_token_to_box
  - trainer: puffer
  - sim: all
  - wandb: metta_research
```

**After (Direct Python):**
```python
from metta import SimpleCNNAgent, Metta

agent = SimpleCNNAgent(hidden_size=256, lstm_layers=3)
metta = Metta(agent=agent)
metta.train()
```

### 2. Modular Components

#### Runtime Configuration (replaces common.yaml)
```python
from metta import configure

runtime = configure(
    run_name="my_experiment",
    device="cuda",
    seed=42,
    data_dir="./experiments"
)
```

#### Agent Creation
```python
from metta import create_agent, SimpleCNNAgent

# Using factory
agent = create_agent("simple_cnn", hidden_size=256)

# Direct instantiation
agent = SimpleCNNAgent(
    obs_width=11,
    obs_height=11,
    hidden_size=256,
    lstm_layers=3
)
```

#### Environment Creation
```python
from metta import create_env, create_env_from_preset

# Custom environment
env = create_env(width=15, height=15, num_agents=2)

# From preset
env = create_env_from_preset("large")
```

### 3. Training Interfaces

#### Minimal Interface
```python
from metta import train

# One-line training
trained_agent = train(agent, timesteps=1_000_000)
```

#### Job Builder Pattern
```python
from metta import JobBuilder

agent = (JobBuilder()
    .with_agent("large_cnn")
    .with_timesteps(5_000_000)
    .with_evaluations("navigation")
    .with_wandb("my_project")
    .run())
```

#### Direct Control
```python
from metta import Metta

metta = Metta(
    agent=agent,
    env=env,
    total_timesteps=10_000_000,
    batch_size=32768
)

while metta.training():
    metta.train(timesteps=100_000)
    results = metta.eval()
    print(f"Reward: {results['mean_reward']}")
```

### 4. Simulation Registry (replaces sim/all.yaml)

```python
from metta import SimulationSpec, register_simulation

# Register custom simulation
register_simulation(
    name="custom/my_eval",
    env="path/to/env",
    num_episodes=10,
    max_time_s=120
)

# Get evaluation suite
suite = get_simulation_suite("navigation")
```

### 5. Component Independence

Each component can be used independently:

```python
# Just create an agent
from metta.agents import SimpleCNNAgent
agent = SimpleCNNAgent(...)

# Just create an environment
from metta.env import create_env
env = create_env(...)

# Just run evaluation
from metta.sim import Simulation
sim = Simulation(agent=agent, env=env)
results = sim.run()
```

## Migration Examples

### From train_job.yaml

**Before:**
```yaml
defaults:
  - common
  - agent: simple_token_to_box
  - trainer: puffer

train_job:
  evals: ${sim}

seed: 1
```

**After:**
```python
from metta import TrainingJob

job = TrainingJob(
    agent="simple_cnn",
    trainer=TrainerConfig(
        total_timesteps=10_000_000,
        batch_size=32768
    ),
    evaluations="all",
    seed=1
)
job.run()
```

### From Agent YAML

**Before:**
```yaml
agent:
  _target_: metta.agent.metta_agent.MettaAgent
  components:
    cnn1:
      _target_: metta.agent.lib.nn_layer_library.Conv2d
      nn_params: {out_channels: 64}
```

**After:**
```python
class MyAgent(BaseAgent):
    def __init__(self, ...):
        super().__init__(...)
        self.cnn1 = nn.Conv2d(in_channels, 64, kernel_size=5)
```

## Best Practices

### 1. Use Type Hints
```python
def create_agent(name: str, hidden_size: int = 128) -> BaseAgent:
    ...
```

### 2. Prefer Composition
```python
# Good: Compose functionality
agent = SimpleCNNAgent(...)
optimizer = torch.optim.Adam(agent.parameters())
metta = Metta(agent=agent, optimizer=optimizer)

# Avoid: Monolithic configs
config = load_yaml("massive_config.yaml")
```

### 3. Make Components Testable
```python
# Each component works standalone
agent = create_agent("simple_cnn")
assert agent.total_params > 0

env = create_env()
assert env.observation_space is not None
```

## Summary of New Interfaces

### Core Imports
```python
from metta import (
    # Runtime
    configure, RuntimeConfig,

    # Agents
    SimpleCNNAgent, create_agent,

    # Environments
    create_env, create_env_from_preset,

    # Training
    train, Metta, TrainingJob, JobBuilder,

    # Simulations
    SimulationSpec, register_simulation
)
```

### Quick Start
```python
# Simplest possible usage
from metta import train, SimpleCNNAgent

agent = SimpleCNNAgent()
trained = train(agent, timesteps=1_000_000)
```

### Advanced Usage
```python
# Full control
from metta import Metta, SimpleCNNAgent, create_env

agent = SimpleCNNAgent(hidden_size=512)
env = create_env(width=21, height=21)

metta = Metta(
    agent=agent,
    env=env,
    optimizer=torch.optim.AdamW(agent.parameters()),
    on_checkpoint=lambda m: m.save(f"ckpt_{m.epoch}.pt")
)

metta.train()
```

## Benefits

1. **No Configuration Files Required**: Everything can be done in Python
2. **IDE Support**: Full autocomplete and type checking
3. **Debugging**: Standard Python debugging tools work
4. **Flexibility**: Mix and match components as needed
5. **Testability**: Each component can be tested in isolation
6. **Gradual Adoption**: Old YAML configs still work

The refactoring makes Metta a true Python library that can be imported and used like any other ML library (PyTorch, scikit-learn, etc.) while maintaining backward compatibility with existing configurations.
