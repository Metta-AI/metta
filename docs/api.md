# Metta API Documentation

The Metta API provides a simple interface for using Metta as a library without complex Hydra configuration management. This allows you to integrate Metta into your own training pipelines and experiments with full control over the training loop.

## Overview

The API provides:
- **Typed configuration classes** for structured, validated settings
- **Direct instantiation functions** for creating agents, environments, optimizers, etc.
- **Training and evaluation functions** for standard RL workflows
- **Full control over the training loop** - no hidden logic or wrappers

## Quick Start

Here's a minimal example to get you started:

```python
import torch
from metta.api import (
    GameConfig, EnvConfig, AgentModelConfig,
    make_environment, make_agent, make_optimizer,
    rollout, train_ppo, eval_policy
)

# Create environment
env_config = EnvConfig(game=GameConfig(num_agents=4, width=32, height=32))
env = make_environment(env_config)

# Create agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = make_agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=AgentModelConfig(hidden_dim=512)
)

# Create optimizer
optimizer = make_optimizer(agent)

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
@dataclass
class OptimizerConfig:
    type: Literal["adam", "muon"] = "adam"   # Optimizer type
    learning_rate: float = 0.000457           # Learning rate
    beta1: float = 0.9                        # Adam beta1
    beta2: float = 0.999                      # Adam beta2
    eps: float = 1e-12                        # Adam epsilon
    weight_decay: float = 0                   # L2 weight decay
```

#### `PPOConfig`
PPO algorithm settings (imported from `metta.rl.trainer_config`).

```python
@dataclass
class PPOConfig:
    # PPO hyperparameters
    clip_coef: float = 0.1                    # Policy clipping coefficient
    ent_coef: float = 0.0021                  # Entropy bonus coefficient
    gae_lambda: float = 0.916                 # GAE lambda
    gamma: float = 0.977                      # Discount factor

    # Training parameters
    max_grad_norm: float = 0.5                # Gradient clipping
    vf_clip_coef: float = 0.1                 # Value function clipping
    vf_coef: float = 0.44                     # Value loss coefficient
    l2_reg_loss_coef: float = 0               # L2 regularization
    l2_init_loss_coef: float = 0              # L2 init loss

    # Normalization
    norm_adv: bool = True                     # Normalize advantages
    clip_vloss: bool = True                   # Clip value loss
    target_kl: Optional[float] = None         # Target KL divergence
```

#### `ExperienceConfig`
Experience collection and batching settings.

```python
@dataclass
class ExperienceConfig:
    batch_size: int = 524288                  # Total batch size
    minibatch_size: int = 16384               # Minibatch for updates
    bptt_horizon: int = 64                    # Sequence length
    update_epochs: int = 1                    # PPO update epochs

    # Performance
    zero_copy: bool = True                    # Zero-copy tensors
    cpu_offload: bool = False                 # Offload to CPU
    async_factor: int = 2                     # Async collection factor
    forward_pass_minibatch_target_size: int = 4096  # Forward pass batch
```

#### `CheckpointConfig`
Checkpointing settings (imported from `metta.rl.trainer_config`).

```python
@dataclass
class CheckpointConfig:
    checkpoint_interval: int = 60             # Checkpoint frequency (seconds)
    wandb_checkpoint_interval: int = 300      # W&B upload frequency
    checkpoint_dir: str = ""                  # Checkpoint directory
```

#### `SimulationConfig`
Evaluation and replay settings (imported from `metta.rl.trainer_config`).

```python
@dataclass
class SimulationConfig:
    evaluate_interval: int = 300              # Evaluation frequency (seconds)
    replay_interval: int = 300                # Replay save frequency
    replay_dir: str = ""                      # Replay directory
```

## API Functions

### Environment Creation

#### `make_environment(env_config=None, **kwargs) -> MettaGridEnv`
Creates a MettaGrid environment.

**Parameters:**
- `env_config`: Optional `EnvConfig` object. If None, uses defaults.
- `**kwargs`: Override specific config values (e.g., `num_agents=8`)

**Returns:** `MettaGridEnv` instance

**Example:**
```python
# Using config object
env_config = EnvConfig(game=GameConfig(num_agents=8))
env = make_environment(env_config)

# Using kwargs
env = make_environment(num_agents=8, width=16, height=16)
```

### Agent Creation

#### `make_agent(observation_space, action_space, global_features, device, config=None) -> MettaAgent`
Creates a neural network agent.

**Parameters:**
- `observation_space`: Environment's observation space
- `action_space`: Environment's action space
- `global_features`: List of global features (usually empty)
- `device`: PyTorch device (CPU/CUDA)
- `config`: Optional `AgentModelConfig`

**Returns:** `MettaAgent` instance

**Example:**
```python
agent = make_agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=torch.device("cuda"),
    config=AgentModelConfig(hidden_dim=256, lstm_layers=2)
)
```

### Training Components

#### `make_optimizer(agent, config=None) -> torch.optim.Optimizer`
Creates an optimizer for training.

**Parameters:**
- `agent`: The agent to optimize
- `config`: Optional `OptimizerConfig`

**Returns:** PyTorch optimizer (Adam or Muon)

**Note:** Muon optimizer is not yet implemented and will fall back to Adam with a warning.

#### `make_curriculum(env_path="/env/mettagrid/simple", scenes=None) -> Curriculum`
Creates a curriculum for training.

**Parameters:**
- `env_path`: Path to environment configuration
- `scenes`: Optional list of scenes for multi-task learning

**Returns:** `Curriculum` instance

**Note:** Multi-scene curriculum is not yet implemented.

#### `make_experience_manager(env, agent, config=None) -> Experience`
Creates an experience buffer for collecting rollouts.

**Parameters:**
- `env`: The environment
- `agent`: The agent
- `config`: Optional `ExperienceConfig`

**Returns:** `Experience` manager instance

### Training Loop Functions

#### `rollout(experience, agent, num_steps=None) -> BatchInfo`
Collects experience by running the agent in the environment.

**Parameters:**
- `experience`: Experience manager
- `agent`: The agent
- `num_steps`: Optional number of steps (defaults to batch size)

**Returns:** `BatchInfo` with rollout statistics

**Note:** This is currently a placeholder implementation.

#### `compute_advantages(experience, gamma=0.977, gae_lambda=0.916) -> torch.Tensor`
Computes advantages using Generalized Advantage Estimation (GAE).

**Parameters:**
- `experience`: Experience manager with collected data
- `gamma`: Discount factor
- `gae_lambda`: GAE lambda parameter

**Returns:** Advantages tensor

**Note:** This is currently a placeholder implementation.

#### `train_ppo(agent, optimizer, experience, ppo_config=None, update_epochs=1) -> Dict[str, float]`
Trains the agent using PPO.

**Parameters:**
- `agent`: The agent to train
- `optimizer`: The optimizer
- `experience`: Experience manager with data
- `ppo_config`: Optional `PPOConfig`
- `update_epochs`: Number of update epochs

**Returns:** Dictionary of training statistics

**Note:** This is currently a placeholder implementation returning zero values.

### Checkpointing

#### `save_checkpoint(agent, path, epoch=0, metadata=None) -> None`
Saves agent checkpoint.

**Parameters:**
- `agent`: The agent to save
- `path`: Directory path to save to
- `epoch`: Training epoch number
- `metadata`: Optional metadata dictionary

**Example:**
```python
save_checkpoint(agent, "./checkpoints", epoch=100,
                metadata={"total_steps": 1000000})
```

#### `load_checkpoint(agent, path) -> Dict[str, Any]`
Loads agent checkpoint.

**Parameters:**
- `agent`: The agent to load into
- `path`: Path to checkpoint file

**Returns:** Checkpoint metadata

**Example:**
```python
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

from metta.api import (
    # Configuration classes
    AgentModelConfig,
    CheckpointConfig,
    EnvConfig,
    ExperienceConfig,
    GameConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    # Functions
    compute_advantages,
    eval_policy,
    make_agent,
    make_curriculum,
    make_environment,
    make_experience_manager,
    make_optimizer,
    rollout,
    save_checkpoint,
    train_ppo,
)

# Set device
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
env = make_environment(env_config)

# Create agent with typed config
agent_config = AgentModelConfig(
    hidden_dim=512,
    lstm_layers=1,
    bptt_horizon=8,
)
agent = make_agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=agent_config,
)

# Create optimizer with typed config
optimizer_config = OptimizerConfig(
    type="adam",
    learning_rate=3e-4,
)
optimizer = make_optimizer(agent, config=optimizer_config)

# Create curriculum
curriculum = make_curriculum("/env/mettagrid/simple")

# Create experience manager with typed config
experience_config = ExperienceConfig(
    batch_size=8192,
    minibatch_size=512,
    bptt_horizon=8,
)
experience = make_experience_manager(env, agent, config=experience_config)

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

### Current Limitations

1. **Placeholder Implementations**: The `rollout()`, `compute_advantages()`, and `train_ppo()` functions are currently placeholder implementations. They return dummy values and need to be connected to the actual training infrastructure.

2. **Muon Optimizer**: The Muon optimizer is not yet implemented. Selecting it will fall back to Adam with a warning.

3. **Multi-Scene Curriculum**: Multi-scene curriculum is not yet implemented. It will fall back to single-task curriculum with a warning.

4. **Config Conversion**: The API internally converts the simplified configuration format to the MettaGrid-specific format using `_convert_to_mettagrid_config()`.

### Design Philosophy

The API is designed to:
- Provide a clean, Pythonic interface without Hydra complexity
- Use typed configurations for clarity and validation
- Give users full control over the training loop
- Be framework-agnostic where possible
- Support both simple experiments and complex training pipelines

### Integration with Existing Code

The API wraps existing Metta components:
- `MettaAgent` for neural network policies
- `MettaGridEnv` for the environment
- `Experience` for rollout collection
- Standard PyTorch optimizers and schedulers

This allows you to leverage the full power of Metta while maintaining a simple interface for common use cases.

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
