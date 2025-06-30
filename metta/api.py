"""
Clean API for Metta - provides direct instantiation without Hydra.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import metta.agent.lib.obs as obs_lib
from metta.agent.lib.observations import ObservationType
from metta.agent.metta_agent import MettaAgent
from metta.env.curriculum import Curriculum, SingleTaskCurriculum
from metta.map.scene import Scene
from metta.mettagrid import MettaGrid
from metta.rl.experience import Experience
from metta.rl.gae import compute_advantages as compute_advantages_gae
from metta.rl.pufferlib import BatchInfo, ExperienceStore, RolloutManager
from metta.rl.trainer_config import OptimizerConfig, PPOConfig

# Object type IDs from mettagrid/src/metta/mettagrid/objects/constants.hpp
# These define the type of object in the environment
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


# ============= Helper Functions (moved before typed configs) =============
def _get_default_objects_config():
    """Get default objects configuration."""
    return [
        {"type": "mine", "name": "mine_red", "color": 0},
        {"type": "generator", "name": "generator_red", "color": 0},
        {"type": "converter", "name": "altar", "color": 1},
        {"type": "wall", "name": "wall", "color": 15},
        {"type": "wall", "name": "block", "color": 14},
    ]


def _get_default_actions_config():
    """Get default actions configuration."""
    return [
        {"type": "null", "key": "forward"},
        {"type": "rotate_absolute", "key": "rotate", "delta": 0},
        {"type": "move_with_rotate", "key": "move", "delta": 0},
        {"type": "harvest", "key": "harvest"},
        {"type": "attack", "key": "attack"},
        {"type": "gift", "key": "gift", "item": "battery.red"},
        {"type": "toggle_use", "key": "use_shield", "use": "shield"},
        {"type": "use", "key": "use_energy_red", "use": "channel_red"},
    ]


def _get_default_env_config():
    """Get default environment configuration as a dictionary."""
    return {
        "game": {
            "max_steps": 1000,
            "time_punishment": -0.0001,
            "episode_lifetime": 10000,
            "num_agents": 64,
            "width": 64,
            "height": 64,
            "observation_width": 11,
            "observation_height": 11,
            "num_observation_tokens": 200,
            "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore.red": 0.01,
                    "battery.red": 0.02,
                    "heart": 1,
                    "ore.red_max": 10,
                    "battery.red_max": 10,
                    "heart_max": 1000,
                },
            },
            "inventory_item_names": ["ore.red", "battery.red", "heart", "laser", "armor"],
            "diversity_bonus": {"enabled": False, "similarity_coef": 0.5, "diversity_coef": 0.5},
            "objects": _get_default_objects_config(),
            "actions": _get_default_actions_config(),
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 64,
                "height": 64,
                "border_width": 2,
                "agents": 64,
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
    }


def _get_runtime_config():
    """Get runtime configuration for experience manager."""
    return DictConfig(
        {
            "forward_pass_minibatch_target_size": 4096,
            "async_factor": 2,
            "trainer": {
                "zero_copy": True,
                "cpu_offload": False,
                "forward_pass_minibatch_target_size": 4096,
                "async_factor": 2,
                "batch_size": 524288,
                "minibatch_size": 16384,
            },
            "vectorization": "serial",
        }
    )


# ============= Typed Configuration Classes =============
@dataclass
class ObjectConfig:
    """Configuration for an object type."""

    type: str
    name: str
    color: int


@dataclass
class ActionConfig:
    """Configuration for an action type."""

    type: str
    key: str
    delta: Optional[int] = None
    item: Optional[str] = None
    use: Optional[str] = None


@dataclass
class GameConfig:
    """Configuration for game mechanics."""

    max_steps: int = 1000
    time_punishment: float = -0.0001
    episode_lifetime: int = 10000
    num_agents: int = 64
    width: int = 64
    height: int = 64
    observation_width: int = 11
    observation_height: int = 11
    num_observation_tokens: int = 200

    # Agent configuration
    default_item_max: int = 50
    heart_max: int = 255
    freeze_duration: int = 10

    # Rewards
    action_failure_penalty: float = 0
    ore_red_reward: float = 0.01
    battery_red_reward: float = 0.02
    heart_reward: float = 1
    ore_red_max: int = 10
    battery_red_max: int = 10
    heart_max_reward: int = 1000

    # Inventory
    inventory_item_names: List[str] = field(
        default_factory=lambda: ["ore.red", "battery.red", "heart", "laser", "armor"]
    )

    # Diversity bonus
    diversity_bonus_enabled: bool = False
    similarity_coef: float = 0.5
    diversity_coef: float = 0.5

    # Groups
    groups: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"agent": {"id": 0, "sprite": 0, "props": {}}})
    reward_sharing_groups: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapBuilderConfig:
    """Configuration for map generation."""

    type: str = "Random"
    width: int = 64
    height: int = 64
    border_width: int = 2
    agents: int = 64
    objects: Dict[str, int] = field(
        default_factory=lambda: {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3}
    )


@dataclass
class EnvConfig:
    """Complete environment configuration."""

    game: GameConfig = field(default_factory=GameConfig)
    objects: List[ObjectConfig] = field(
        default_factory=lambda: [
            ObjectConfig(type="mine", name="mine_red", color=0),
            ObjectConfig(type="generator", name="generator_red", color=0),
            ObjectConfig(type="converter", name="altar", color=1),
            ObjectConfig(type="wall", name="wall", color=15),
            ObjectConfig(type="wall", name="block", color=14),
        ]
    )
    actions: List[ActionConfig] = field(
        default_factory=lambda: [
            ActionConfig(type="null", key="forward"),
            ActionConfig(type="rotate_absolute", key="rotate", delta=0),
            ActionConfig(type="move_with_rotate", key="move", delta=0),
            ActionConfig(type="harvest", key="harvest"),
            ActionConfig(type="attack", key="attack"),
            ActionConfig(type="gift", key="gift", item="battery.red"),
            ActionConfig(type="toggle_use", key="use_shield", use="shield"),
            ActionConfig(type="use", key="use_energy_red", use="channel_red"),
        ]
    )
    map_builder: MapBuilderConfig = field(default_factory=MapBuilderConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by MettaGrid."""
        return {
            "game": {
                "max_steps": self.game.max_steps,
                "time_punishment": self.game.time_punishment,
                "episode_lifetime": self.game.episode_lifetime,
                "num_agents": self.game.num_agents,
                "width": self.game.width,
                "height": self.game.height,
                "observation_width": self.game.observation_width,
                "observation_height": self.game.observation_height,
                "num_observation_tokens": self.game.num_observation_tokens,
                "groups": self.game.groups,
                "agent": {
                    "default_item_max": self.game.default_item_max,
                    "heart_max": self.game.heart_max,
                    "freeze_duration": self.game.freeze_duration,
                    "rewards": {
                        "action_failure_penalty": self.game.action_failure_penalty,
                        "ore.red": self.game.ore_red_reward,
                        "battery.red": self.game.battery_red_reward,
                        "heart": self.game.heart_reward,
                        "ore.red_max": self.game.ore_red_max,
                        "battery.red_max": self.game.battery_red_max,
                        "heart_max": self.game.heart_max_reward,
                    },
                },
                "inventory_item_names": self.game.inventory_item_names,
                "diversity_bonus": {
                    "enabled": self.game.diversity_bonus_enabled,
                    "similarity_coef": self.game.similarity_coef,
                    "diversity_coef": self.game.diversity_coef,
                },
                "objects": [vars(obj) for obj in self.objects],
                "actions": [vars(action) for action in self.actions],
                "reward_sharing": {"groups": self.game.reward_sharing_groups},
                "map_builder": {
                    "_target_": f"metta.mettagrid.room.random.{self.map_builder.type}",
                    "width": self.map_builder.width,
                    "height": self.map_builder.height,
                    "border_width": self.map_builder.border_width,
                    "agents": self.map_builder.agents,
                    "objects": self.map_builder.objects,
                },
            },
        }


@dataclass
class AgentModelConfig:
    """Configuration for the ML agent model architecture."""

    hidden_dim: int = 1024
    lstm_layers: int = 1
    use_prev_action: bool = True
    use_prev_reward: bool = True
    mlp_layers: int = 2
    bptt_horizon: int = 8
    forward_lstm: bool = True
    backbone: str = "cnn"
    dtypes_fp16: bool = False
    obs_scale: int = 1
    obs_process_func: str = "flatten_obs_dict"
    observation_types: List[ObservationType] = field(default_factory=lambda: ["entities_state"])
    clip_range: float = 0
    analyze_weights_interval: int = 300
    l2_init_weight_update_interval: int = 0


@dataclass
class ExperienceConfig:
    """Configuration for experience collection."""

    batch_size: int = 524288
    minibatch_size: int = 16384
    bptt_horizon: int = 64
    update_epochs: int = 1
    zero_copy: bool = True
    cpu_offload: bool = False
    async_factor: int = 2
    forward_pass_minibatch_target_size: int = 4096


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""

    epoch: int
    agent_step: int
    total_agent_step: int
    optimizer_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Optional[Dict[str, Any]]
    policy_path: Optional[str]
    stopwatch_state: Optional[Dict[str, Any]]
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def save(self, checkpoint_dir: str) -> str:
        """Save training state to checkpoint file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"training_state_epoch_{self.epoch}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load training state from checkpoint file."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ============= API Functions =============
def make_environment(
    env_config: Optional[EnvConfig] = None,
    **kwargs,
) -> MettaGrid:
    """Create a MettaGrid environment.

    Args:
        env_config: EnvConfig object. If None, uses defaults.
        **kwargs: Additional keyword arguments to override config values.

    Returns:
        MettaGrid environment instance.
    """
    config = env_config or EnvConfig()

    # Apply kwargs overrides
    if "num_agents" in kwargs:
        config.game.num_agents = kwargs["num_agents"]
        config.map_builder.agents = kwargs["num_agents"]
    if "width" in kwargs:
        config.game.width = kwargs["width"]
        config.map_builder.width = kwargs["width"]
    if "height" in kwargs:
        config.game.height = kwargs["height"]
        config.map_builder.height = kwargs["height"]
    if "max_steps" in kwargs:
        config.game.max_steps = kwargs["max_steps"]

    # Create environment
    env = MettaGrid(config.to_dict()["game"], None, 0)
    env.enable_history(True)
    return env


def make_agent(
    observation_space: Any,
    action_space: Any,
    global_features: List[str],
    device: torch.device,
    config: Optional[AgentModelConfig] = None,
) -> MettaAgent:
    """Create a Metta agent.

    Args:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        global_features: List of global features.
        device: Torch device to use.
        config: AgentModelConfig object. If None, uses defaults.

    Returns:
        MettaAgent instance.
    """
    config = config or AgentModelConfig()

    # Create observation processor
    obs_preprocessor = obs_lib.make_obs_preprocessor(
        observation_space,
        global_features,
        config.observation_types,
        1,  # lstm_layers
        device,
        config.obs_scale,
        config.obs_process_func,
    )

    # Build components config
    components = [
        ("hidden_0", {"_target_": "metta.agent.components.mlp.MLP", "layer_width": config.hidden_dim}),
        (
            "recurrent",
            {
                "_target_": "metta.agent.lib.lstm.FastLSTM",
                "input_size": config.hidden_dim,
                "hidden_size": config.hidden_dim,
                "num_layers": config.lstm_layers,
                "batch_first": True,
            },
        ),
    ]

    # Add critic layers
    for i in range(config.mlp_layers):
        components.append(
            (f"critic_{i}", {"_target_": "metta.agent.components.mlp.MLP", "layer_width": config.hidden_dim})
        )

    # Configure agent
    agent_config = DictConfig(
        {
            "observation_space": observation_space,
            "action_space": action_space,
            "global_features": global_features,
            "hidden_dim": config.hidden_dim,
            "lstm_layers": config.lstm_layers,
            "use_prev_action": config.use_prev_action,
            "use_prev_reward": config.use_prev_reward,
            "mlp_layers": config.mlp_layers,
            "bptt_horizon": config.bptt_horizon,
            "clip_range": config.clip_range,
            "forward_lstm": config.forward_lstm,
            "backbone": config.backbone,
            "analyze_weights_interval": config.analyze_weights_interval,
            "l2_init_weight_update_interval": config.l2_init_weight_update_interval,
            "components": components,
            "dtypes_fp16": config.dtypes_fp16,
            "obs_preprocessor": obs_preprocessor,
            "device": device,
        }
    )

    return MettaAgent(agent_config)


def make_optimizer(
    agent: MettaAgent,
    config: Optional[OptimizerConfig] = None,
) -> torch.optim.Optimizer:
    """Create an optimizer for training.

    Args:
        agent: The agent to optimize.
        config: OptimizerConfig object. If None, uses defaults.

    Returns:
        Torch optimizer instance.
    """
    config = config or OptimizerConfig()

    if config.type == "adam":
        return torch.optim.Adam(
            agent.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.type == "muon":
        from metta.rl.muon import Muon

        return Muon(
            agent.parameters(),
            lr=config.learning_rate,
            momentum=config.beta1,
            nesterov=True,
            ns_steps=6,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")


def make_curriculum(
    env_path: str = "/env/mettagrid/simple",
    scenes: Optional[List[Scene]] = None,
) -> Curriculum:
    """Create a curriculum for training.

    Args:
        env_path: Path to the environment configuration.
        scenes: Optional list of scenes. If None, creates single task curriculum.

    Returns:
        Curriculum instance.
    """
    if scenes is None:
        return SingleTaskCurriculum(env_path, {})
    else:
        from metta.env.curriculum import MultiSceneCurriculum

        return MultiSceneCurriculum(scenes, {})


def make_experience_manager(
    env: MettaGrid,
    agent: MettaAgent,
    config: Optional[ExperienceConfig] = None,
) -> Experience:
    """Create an experience manager for collecting rollouts.

    Args:
        env: The environment.
        agent: The agent.
        config: ExperienceConfig object. If None, uses defaults.

    Returns:
        Experience manager instance.
    """
    config = config or ExperienceConfig()

    # Create runtime config
    cfg = DictConfig(
        {
            "forward_pass_minibatch_target_size": config.forward_pass_minibatch_target_size,
            "async_factor": config.async_factor,
            "trainer": {
                "zero_copy": config.zero_copy,
                "cpu_offload": config.cpu_offload,
                "forward_pass_minibatch_target_size": config.forward_pass_minibatch_target_size,
                "async_factor": config.async_factor,
                "batch_size": config.batch_size,
                "minibatch_size": config.minibatch_size,
                "bptt_horizon": config.bptt_horizon,
            },
            "vectorization": "serial",
        }
    )

    # Create experience store
    experience_store = ExperienceStore(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        agents_per_batch=env.num_agents,
        max_steps_per_episode=env.max_steps,
        config=cfg,
        global_features=[],
    )

    # Create rollout manager
    rollout_manager = RolloutManager(
        env=env,
        agent=agent,
        experience_store=experience_store,
        cfg=cfg,
    )

    # Create experience manager
    return Experience(
        env=env,
        rollout_manager=rollout_manager,
        experience_store=experience_store,
        device=agent.device,
        minibatch_size=config.minibatch_size,
        batch_size=config.batch_size,
        bptt_horizon=config.bptt_horizon,
        cpu_offload=config.cpu_offload,
        cfg=cfg,
    )


# Training functions
def rollout(
    experience: Experience,
    agent: MettaAgent,
    num_steps: Optional[int] = None,
) -> BatchInfo:
    """Collect experience by running the agent in the environment.

    Args:
        experience: Experience manager.
        agent: The agent to run.
        num_steps: Number of steps to collect. If None, collects a full batch.

    Returns:
        BatchInfo containing rollout statistics.
    """
    return experience.rollout(agent, num_steps or experience.batch_size)


def compute_advantages(
    experience: Experience,
    gamma: float = 0.977,
    gae_lambda: float = 0.916,
) -> torch.Tensor:
    """Compute advantages using GAE.

    Args:
        experience: Experience manager containing collected data.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        Advantages tensor.
    """
    batch = experience.get_batch()
    return compute_advantages_gae(
        batch["rewards"],
        batch["values"],
        batch["dones"],
        gamma,
        gae_lambda,
    )


def train_ppo(
    agent: MettaAgent,
    optimizer: torch.optim.Optimizer,
    experience: Experience,
    ppo_config: Optional[PPOConfig] = None,
    update_epochs: int = 1,
) -> Dict[str, float]:
    """Train the agent using PPO.

    Args:
        agent: The agent to train.
        optimizer: The optimizer.
        experience: Experience manager with collected data.
        ppo_config: PPOConfig object. If None, uses defaults.
        update_epochs: Number of epochs to train.

    Returns:
        Dictionary of training statistics.
    """
    config = ppo_config or PPOConfig()

    # Placeholder for actual PPO implementation
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }


def save_checkpoint(
    agent: MettaAgent,
    path: str,
    epoch: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save agent checkpoint.

    Args:
        agent: The agent to save.
        path: Directory path to save to.
        epoch: Training epoch number.
        metadata: Optional metadata to include.
    """
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, f"checkpoint_{epoch:06d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": agent.state_dict(),
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def load_checkpoint(agent: MettaAgent, path: str) -> Dict[str, Any]:
    """Load agent checkpoint.

    Args:
        agent: The agent to load into.
        path: Path to checkpoint file.

    Returns:
        Checkpoint metadata.
    """
    checkpoint = torch.load(path, map_location=agent.device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})


def eval_policy(
    agent: MettaAgent,
    env: MettaGrid,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate a policy.

    Args:
        agent: The agent to evaluate.
        env: The environment.
        num_episodes: Number of episodes to run.

    Returns:
        Dictionary of evaluation statistics.
    """
    total_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        hidden_state = agent.initial_hidden_state(env.num_agents)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(agent.device)
                action, _, _, hidden_state = agent(obs_tensor, hidden_state)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward.sum()
            episode_length += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


# Helper functions
def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    batch_size: int,
    warmup_steps: Optional[int] = None,
    schedule_type: str = "linear",
    anneal_lr: bool = True,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler."""
    if not anneal_lr:
        return None

    total_updates = total_timesteps // batch_size

    if schedule_type == "linear":

        def lr_lambda(epoch):
            if warmup_steps and epoch < warmup_steps:
                return epoch / warmup_steps
            if total_updates <= (warmup_steps or 0):
                return 1.0
            progress = (epoch - (warmup_steps or 0)) / (total_updates - (warmup_steps or 0))
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)
    elif schedule_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_updates)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    grad_norms = []
    param_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            param_norms.append(param.norm().item())

    if not grad_norms:
        return {}

    return {
        "grad_norm_mean": np.mean(grad_norms),
        "grad_norm_max": np.max(grad_norms),
        "grad_norm_min": np.min(grad_norms),
        "param_norm_mean": np.mean(param_norms),
    }
