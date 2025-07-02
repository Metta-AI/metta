# Metta API Documentation

The Metta API provides a simple interface for using Metta as a library without complex Hydra configuration management. This allows you to integrate Metta into your own training pipelines and experiments with full control over the training loop.

## Overview

The API provides:
- **Typed configuration classes** for structured, validated settings
- **Constructor-style factory classes** for creating agents, environments, optimizers, etc.
- **Training and evaluation functions** for standard RL workflows
- **Full control over the training loop** - no hidden logic or wrappers

## Quick Start

Here's a minimal example to get you started:

```python
import torch
from metta.api import (
    GameConfig, EnvConfig, AgentModelConfig,
    Environment, Agent, Optimizer,
    rollout, train_ppo, eval_policy
)

# Create environment
env_config = EnvConfig(game=GameConfig(num_agents=4, width=32, height=32))
env = Environment(config=env_config)

# Create agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=AgentModelConfig(hidden_dim=512)
)

# Create optimizer
optimizer = Optimizer(agent)

# Evaluate untrained policy
stats = eval_policy(agent, env, num_episodes=5)
print(f"Initial performance: {stats}")
```

## Configuration Classes

The API uses typed dataclasses for configuration, providing validation and clear documentation of available options.

### Environment Configuration

#### `ObjectConfig`
Defines an object type in the environment.

```python
@dataclass
class ObjectConfig:
    type: str       # Object type: "mine", "generator", "converter", "wall"
    name: str       # Unique name for the object
    color: int      # Color index (0-15)
```

#### `ActionConfig`
Defines an action that agents can perform.

```python
@dataclass
class ActionConfig:
    type: str                    # Action type
    key: str                     # Action key/name
    delta: Optional[int] = None  # For rotation actions
    item: Optional[str] = None   # For gift actions
    use: Optional[str] = None    # For use actions
```

#### `GameConfig`
Core game mechanics configuration.

```python
@dataclass
class GameConfig:
    # Episode settings
    max_steps: int = 1000                    # Max steps per episode
    time_punishment: float = -0.0001         # Penalty per timestep
    episode_lifetime: int = 10000            # Total episode lifetime

    # World settings
    num_agents: int = 64                     # Number of agents
    width: int = 64                          # World width
    height: int = 64                         # World height

    # Observation settings
    observation_width: int = 11              # Agent observation width
    observation_height: int = 11             # Agent observation height
    num_observation_tokens: int = 200        # Max observation tokens

    # Agent settings
    default_item_max: int = 50               # Max items per type
    heart_max: int = 255                     # Max hearts
    freeze_duration: int = 10                # Freeze duration when hit

    # Reward settings
    action_failure_penalty: float = 0        # Penalty for invalid actions
    ore_red_reward: float = 0.01             # Reward for collecting red ore
    battery_red_reward: float = 0.02         # Reward for red batteries
    heart_reward: float = 1                  # Reward for hearts
    ore_red_max: int = 10                    # Max red ore rewards
    battery_red_max: int = 10                # Max battery rewards
    heart_max_reward: int = 1000             # Max heart rewards

    # Inventory
    inventory_item_names: List[str]          # Available inventory items

    # Diversity bonus
    diversity_bonus_enabled: bool = False    # Enable diversity rewards
    similarity_coef: float = 0.5             # Similarity coefficient
    diversity_coef: float = 0.5              # Diversity coefficient

    # Agent groups
    groups: Dict[str, Dict[str, Any]]        # Agent group definitions
    reward_sharing_groups: Dict[str, Any]    # Reward sharing config
```

#### `MapBuilderConfig`
Configuration for procedural map generation.

```python
@dataclass
class MapBuilderConfig:
    type: str = "Random"                     # Map builder type
    width: int = 64                          # Map width
    height: int = 64                         # Map height
    border_width: int = 2                    # Border thickness
    agents: int = 64                         # Number of agent spawns
    objects: Dict[str, int]                  # Object counts by type
    # Default: {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3}
```

#### `EnvConfig`
Complete environment configuration combining all the above.

```python
@dataclass
class EnvConfig:
    game: GameConfig                         # Game mechanics
    objects: List[ObjectConfig]              # Available objects
    actions: List[ActionConfig]              # Available actions
    map_builder: MapBuilderConfig            # Map generation config
```

### Agent Configuration

#### `AgentModelConfig`
Neural network architecture configuration.

```python
@dataclass
class AgentModelConfig:
    # Architecture
    hidden_dim: int = 1024                   # LSTM hidden dimension
    lstm_layers: int = 1                     # Number of LSTM layers
    mlp_layers: int = 2                      # MLP layers for heads
    backbone: str = "cnn"                    # Feature extractor type

    # Training settings
    bptt_horizon: int = 8                    # Backprop through time horizon
    forward_lstm: bool = True                # Use forward LSTM pass
    clip_range: float = 0                    # Gradient clipping range

    # Input features
    use_prev_action: bool = True             # Include previous action
    use_prev_reward: bool = True             # Include previous reward
    observation_types: List[str]             # Observation types to use

    # Performance
    dtypes_fp16: bool = False                # Use half precision
    obs_scale: int = 1                       # Observation scaling
    obs_process_func: str = "flatten_obs_dict"  # Observation processing

    # Analysis
    analyze_weights_interval: int = 300      # Weight analysis frequency
    l2_init_weight_update_interval: int = 0  # L2 regularization interval
```

### Training Configuration

#### `OptimizerConfig`
Optimizer settings (imported from `metta.rl.trainer_config`).

```python
from metta.rl.trainer_config import OptimizerConfig

config = OptimizerConfig(
    type="adam",  # or "muon"
    learning_rate=0.0004,
    beta1=0.9,
    beta2=0.999,
    eps=1e-12,
    weight_decay=0
)
```

#### `PPOConfig`
PPO algorithm configuration (imported from `metta.rl.trainer_config`).

```python
from metta.rl.trainer_config import PPOConfig

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

#### `ExperienceConfig`
Experience collection configuration.

```python
@dataclass
class ExperienceConfig:
    batch_size: int = 524288
    minibatch_size: int = 16384
    bptt_horizon: int = 64
    update_epochs: int = 1
    zero_copy: bool = True
    cpu_offload: bool = False
    async_factor: int = 2
    forward_pass_minibatch_target_size: int = 4096
```

#### `CheckpointConfig`
Checkpoint configuration (imported from `metta.rl.trainer_config`).

```python
from metta.rl.trainer_config import CheckpointConfig

config = CheckpointConfig(
    checkpoint_interval=60,
    wandb_checkpoint_interval=300,
    checkpoint_dir="./checkpoints"
)
```

#### `SimulationConfig`
Simulation/evaluation configuration (imported from `metta.rl.trainer_config`).

```python
from metta.rl.trainer_config import SimulationConfig

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
print(f"Loaded checkpoint from epoch {metadata.get('epoch', 0)}")
```

### Evaluation

#### `eval_policy(agent, env, num_episodes=10) -> Dict[str, float]`
Evaluates a trained policy.

**Parameters:**
- `agent`: The agent to evaluate
- `env`: The environment
- `num_episodes`: Number of episodes to run

**Returns:** Dictionary with evaluation statistics:
- `mean_reward`: Average total reward
- `std_reward`: Standard deviation of rewards
- `mean_length`: Average episode length
- `std_length`: Standard deviation of lengths

### Helper Functions

#### `make_lr_scheduler(optimizer, total_timesteps, batch_size, warmup_steps=None, schedule_type="linear", anneal_lr=True)`
Creates a learning rate scheduler.

**Parameters:**
- `optimizer`: The optimizer
- `total_timesteps`: Total training timesteps
- `batch_size`: Batch size (for computing updates)
- `warmup_steps`: Optional warmup period
- `schedule_type`: "linear" or "cosine"
- `anneal_lr`: Whether to anneal (if False, returns None)

**Returns:** Optional LR scheduler

#### `compute_gradient_stats(model) -> Dict[str, float]`
Computes gradient statistics for monitoring.

**Parameters:**
- `model`: The neural network model

**Returns:** Dictionary with gradient statistics:
- `grad_norm_mean`: Mean gradient norm
- `grad_norm_max`: Maximum gradient norm
- `grad_norm_min`: Minimum gradient norm
- `param_norm_mean`: Mean parameter norm

### Additional Classes

#### `BatchInfo`
Information about collected experience batches.

```python
@dataclass
class BatchInfo:
    total_env_steps: int                      # Total environment steps
    episode_returns: Optional[List[float]]    # Episode returns
    episode_lengths: Optional[List[int]]      # Episode lengths
```

#### `TrainingState`
Complete training state for advanced checkpointing.

```python
@dataclass
class TrainingState:
    epoch: int                                # Current epoch
    agent_step: int                           # Agent steps
    total_agent_step: int                     # Total agent steps
    optimizer_state_dict: Dict[str, Any]      # Optimizer state
    lr_scheduler_state_dict: Optional[Dict]   # LR scheduler state
    policy_path: Optional[str]                # Policy save path
    stopwatch_state: Optional[Dict]           # Timing information
    extra_args: Dict[str, Any]                # Additional data

    def save(self, checkpoint_dir: str) -> str:
        """Save training state to file."""

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load training state from file."""
```

## Complete Example

Here's a complete training example that demonstrates all the key components:

```python
"""Example of using Metta as a library without Hydra configuration."""

import torch
from omegaconf import DictConfig
from metta.api import (
    # Configuration classes
    Agent,
    AgentModelConfig,
    CheckpointConfig,
    Environment,
    EnvConfig,
    ExperienceConfig,
    ExperienceManager,
    GameConfig,
    Optimizer,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    # Functions
    compute_advantages,
    eval_policy,
    rollout,
    save_checkpoint,
    train_ppo,
)
from metta.mettagrid.curriculum.core import SingleTaskCurriculum

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment with typed config
game_config = GameConfig(
    max_steps=1000,
    num_agents=4,
    width=32,
    height=32,
)
env_config = EnvConfig(game=game_config)
env = Environment(config=env_config)

# Create agent with typed config
agent_config = AgentModelConfig(
    hidden_dim=512,
    lstm_layers=1,
    bptt_horizon=8,
)
agent = Agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=agent_config,
)

# Create optimizer
optimizer = Optimizer(agent, learning_rate=3e-4)

# Create curriculum directly
curriculum = SingleTaskCurriculum("/env/mettagrid/simple", DictConfig({}))

# Create experience manager
experience = ExperienceManager(env, agent, batch_size=8192, minibatch_size=512)

# Training configuration
ppo_config = PPOConfig(
    clip_coef=0.1,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
)

checkpoint_config = CheckpointConfig(
    checkpoint_interval=100,
    checkpoint_dir="./checkpoints",
)

simulation_config = SimulationConfig(
    evaluate_interval=500,
)

# Training loop
total_steps = 0
epoch = 0
target_steps = 100_000

print(f"Starting training for {target_steps} steps...")

while total_steps < target_steps:
    # Collect experience
    print(f"Epoch {epoch}: Collecting rollouts...")
    batch_info = rollout(experience, agent)
    total_steps += batch_info.total_env_steps

    # Compute advantages
    advantages = compute_advantages(
        experience,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda
    )

    # Train
    print(f"Epoch {epoch}: Training PPO...")
    train_stats = train_ppo(
        agent,
        optimizer,
        experience,
        ppo_config=ppo_config,
        update_epochs=4,
    )

    # Print stats
    print(f"Epoch {epoch}: Steps={total_steps}, Stats={train_stats}")

    # Save checkpoint
    if epoch % checkpoint_config.checkpoint_interval == 0:
        save_checkpoint(agent, checkpoint_config.checkpoint_dir, epoch)
        print(f"Saved checkpoint at epoch {epoch}")

    # Evaluate
    if epoch % simulation_config.evaluate_interval == 0 and epoch > 0:
        print("Evaluating policy...")
        eval_stats = eval_policy(agent, env, num_episodes=5)
        print(f"Evaluation stats: {eval_stats}")

    epoch += 1

print("Training complete!")

# Final evaluation
print("Running final evaluation...")
final_stats = eval_policy(agent, env, num_episodes=10)
print(f"Final evaluation results: {final_stats}")
```

## Object Type Constants

The API provides constants for object types used in the environment:

```python
# Object type IDs from mettagrid/src/metta/mettagrid/objects/constants.hpp
# TODO: These should be imported from mettagrid once they're exposed via Python bindings
TYPE_AGENT = 0
TYPE_WALL = 1
TYPE_MINE_RED = 2
TYPE_MINE_BLUE = 3
TYPE_MINE_GREEN = 4
TYPE_GENERATOR_RED = 5
TYPE_GENERATOR_BLUE = 6
TYPE_GENERATOR_GREEN = 7
TYPE_ALTAR = 8
TYPE_ARMORY = 9
TYPE_LASERY = 10
TYPE_LAB = 11
TYPE_FACTORY = 12
TYPE_TEMPLE = 13
TYPE_GENERIC_CONVERTER = 14
```

## Implementation Notes

1. **Placeholder Implementations**: The `rollout()`, `compute_advantages()`, and `train_ppo()` functions are currently placeholder implementations. They return dummy values and need to be connected to the actual training infrastructure.

2. **Muon Optimizer**: The Muon optimizer is not yet implemented. Selecting it will fall back to Adam with a warning.

3. **Multi-Scene Curriculum**: Multi-scene curriculum is not yet implemented. Creating a curriculum with scenes will show a warning.

4. **Config Conversion**: The API internally converts the simplified configuration format to the MettaGrid-specific format using `_convert_to_mettagrid_config()`.

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
from metta.mettagrid.mettagrid_env import MettaGridEnv
env = MettaGridEnv(curriculum=curriculum, render_mode=None)
```

## Backward Compatibility

For backward compatibility, the old `make_` functions are still available but deprecated:

```python
# Deprecated functions (use the factory classes instead)
from metta.api import make_environment, make_agent, make_optimizer, make_experience_manager

# Old way (deprecated)
env = make_environment(config)
agent = make_agent(obs_space, action_space, [], device, config)

# New way (recommended)
env = Environment(config=config)
agent = Agent(obs_space, action_space, [], device, config=config)
```

These deprecated functions will show a warning when used and may be removed in future versions.
