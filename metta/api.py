"""
Clean API for Metta - provides direct instantiation without Hydra.

This API exposes the core training components from Metta, allowing users to:
1. Create environments, agents, and training components without Hydra
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.common.profiling.stopwatch import Stopwatch
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functions import (
    accumulate_rollout_stats,
    compute_advantage,
    get_lstm_config,
    perform_rollout_step,
    process_minibatch_update,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.kickstarter_config import KickstartConfig
from metta.rl.losses import Losses
from metta.rl.trainer_config import (
    CheckpointConfig,
    OptimizerConfig,
    PPOConfig,
    PrioritizedExperienceReplayConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
    VTraceConfig,
)
from metta.rl.vecenv import make_vecenv

logger = logging.getLogger(__name__)

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


# Helper to create default environment config
def _get_default_env_config(num_agents: int = 4, width: int = 32, height: int = 32) -> Dict[str, Any]:
    """Get default environment configuration."""
    return {
        "game": {
            "max_steps": 1000,
            "num_agents": num_agents,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "inventory_item_names": ["ore_red", "battery_red", "heart", "laser", "armor"],
            "groups": {"agent": {"id": 0, "sprite": 0}},
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore_red": 0.01,
                    "battery_red": 0.02,
                    "heart": 1,
                    "ore_red_max": 10,
                    "battery_red_max": 10,
                    "heart_max": 1000,
                },
            },
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
            },
            "objects": {
                "mine_red": {
                    "type_id": TYPE_MINE_RED,
                    "output_ore_red": 1,
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 0,
                    "initial_items": 0,
                    "color": 0,
                },
                "generator_red": {
                    "type_id": TYPE_GENERATOR_RED,
                    "input_ore_red": 1,
                    "output_battery_red": 1,
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 0,
                    "initial_items": 0,
                    "color": 0,
                },
                "altar": {
                    "type_id": TYPE_ALTAR,
                    "input_battery_red": 3,
                    "output_heart": 1,
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 0,
                    "initial_items": 0,
                    "color": 1,
                },
                "wall": {"type_id": TYPE_WALL, "swappable": False},
                "block": {"type_id": 14, "swappable": True},  # Different type_id for block
            },
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": width,
                "height": height,
                "border_width": 2,
                "agents": num_agents,
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
    }


class Environment:
    """Factory for creating MettaGrid environments with a clean API.

    This wraps the environment creation process, handling curriculum setup
    and configuration without requiring Hydra.

    Note: This returns a vecenv (vectorized environment) wrapper, not an
    Environment instance. The vecenv has methods like reset(), step(), close().
    """

    def __new__(
        cls,
        curriculum_path: Optional[str] = None,
        env_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        seed: Optional[int] = None,
        num_envs: int = 1,
        num_workers: int = 1,
        batch_size: int = 1,
        async_factor: int = 1,
        zero_copy: bool = True,
        is_training: bool = True,
        vectorization: str = "serial",
        # Convenience parameters for quick setup
        num_agents: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Any:  # Returns pufferlib vecenv wrapper
        """Create a vectorized MettaGrid environment.

        Args:
            curriculum_path: Optional path to environment configuration (e.g., "/env/mettagrid/simple")
            env_config: Optional complete environment config dict. If not provided, uses defaults.
            device: Device to use
            seed: Random seed
            num_envs: Number of parallel environments
            num_workers: Number of worker processes
            batch_size: Batch size for environment steps
            async_factor: Async factor for environment
            zero_copy: Whether to use zero-copy optimization
            is_training: Whether this is for training
            vectorization: Vectorization mode
            num_agents: Convenience parameter to set number of agents
            width: Convenience parameter to set environment width
            height: Convenience parameter to set environment height

        Returns:
            Vectorized environment wrapper with reset(), step(), close() methods
        """
        # Create config if not provided
        if env_config is None:
            # Use convenience parameters if provided
            env_config = _get_default_env_config(
                num_agents=num_agents or 4,
                width=width or 32,
                height=height or 32,
            )
        else:
            # Apply convenience parameter overrides to provided config
            if num_agents is not None:
                env_config["game"]["num_agents"] = num_agents
                if "map_builder" in env_config["game"]:
                    env_config["game"]["map_builder"]["agents"] = num_agents
            if width is not None:
                if "map_builder" in env_config["game"]:
                    env_config["game"]["map_builder"]["width"] = width
            if height is not None:
                if "map_builder" in env_config["game"]:
                    env_config["game"]["map_builder"]["height"] = height

        # Create curriculum
        task_config = DictConfig(env_config)
        curriculum_name = curriculum_path or "custom_env"
        curriculum = SingleTaskCurriculum(curriculum_name, task_config)

        # Create vectorized environment
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization=vectorization,
            num_envs=num_envs,
            batch_size=batch_size,
            num_workers=num_workers,
            zero_copy=zero_copy,
            is_training=is_training,
        )

        # Set seed
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,)).item())
        vecenv.async_reset(seed)

        return vecenv


class Agent:
    """Factory for creating Metta agents with a clean API.

    This handles agent creation and initialization without Hydra.
    """

    def __new__(
        cls,
        env: Any,  # vecenv wrapper
        config: DictConfig,
        device: str = "cuda",
    ) -> MettaAgent:
        """Create a Metta agent.

        Args:
            env: Vectorized environment (from Environment factory)
            config: DictConfig with agent configuration
            device: Device to use

        Returns:
            MettaAgent instance
        """
        # Get the actual MettaGridEnv from vecenv wrapper
        metta_grid_env = env.driver_env
        assert isinstance(metta_grid_env, MettaGridEnv)

        # Create observation space matching what make_policy does
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": metta_grid_env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Extract agent config
        agent_cfg = config.agent

        # Create MettaAgent directly without Hydra
        agent = MettaAgent(
            obs_space=obs_space,
            obs_width=metta_grid_env.obs_width,
            obs_height=metta_grid_env.obs_height,
            action_space=metta_grid_env.single_action_space,
            feature_normalizations=metta_grid_env.feature_normalizations,
            global_features=metta_grid_env.global_features,
            device=str(device),
            **agent_cfg,
        )

        # Initialize to environment
        features = metta_grid_env.get_observation_features()
        agent.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)

        return agent


class TrainingComponents:
    """Container for all components needed for training.

    This exposes the internal components that MettaTrainer uses,
    allowing direct control over the training loop.
    """

    def __init__(
        self,
        vecenv: Any,
        policy: MettaAgent,
        experience: Experience,
        optimizer: torch.optim.Optimizer,
        losses: Losses,
        kickstarter: Kickstarter,
        timer: Stopwatch,
        trainer_config: TrainerConfig,
        device: torch.device,
    ):
        self.vecenv = vecenv
        self.policy = policy
        self.experience = experience
        self.optimizer = optimizer
        self.losses = losses
        self.kickstarter = kickstarter
        self.timer = timer
        self.trainer_config = trainer_config
        self.device = device

        # Training state
        self.agent_step = 0
        self.epoch = 0
        self.stats = {}

    @classmethod
    def create(
        cls,
        vecenv: Any,
        policy: MettaAgent,
        trainer_config: TrainerConfig,
        device: str = "cuda",
        policy_store: Optional[PolicyStore] = None,
    ) -> "TrainingComponents":
        """Create all training components.

        Args:
            vecenv: Vectorized environment
            policy: Agent policy
            trainer_config: Training configuration
            device: Device to use
            policy_store: Optional policy store for kickstarter

        Returns:
            TrainingComponents instance with all components initialized
        """
        device_obj = torch.device(device)

        # Get environment info
        metta_grid_env = vecenv.driver_env
        assert isinstance(metta_grid_env, MettaGridEnv)

        # Create optimizer
        if trainer_config.optimizer.type == "adam":
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=trainer_config.optimizer.learning_rate,
                betas=(trainer_config.optimizer.beta1, trainer_config.optimizer.beta2),
                eps=trainer_config.optimizer.eps,
                weight_decay=trainer_config.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {trainer_config.optimizer.type}")

        # Create experience buffer
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        hidden_size, num_lstm_layers = get_lstm_config(policy)

        experience = Experience(
            total_agents=total_agents,
            batch_size=trainer_config.batch_size,
            bptt_horizon=trainer_config.bptt_horizon,
            minibatch_size=trainer_config.minibatch_size,
            max_minibatch_size=trainer_config.minibatch_size,
            obs_space=obs_space,
            atn_space=atn_space,
            device=device_obj,
            hidden_size=hidden_size,
            cpu_offload=trainer_config.cpu_offload,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

        # Create kickstarter
        if policy_store is not None:
            kickstarter = Kickstarter(
                trainer_config.kickstart,
                str(device_obj),
                policy_store,
                metta_grid_env.action_names,
                metta_grid_env.max_action_args,
            )
        else:
            # Create a dummy kickstarter if no policy store is provided
            # This is fine since kickstart will be disabled anyway
            raise ValueError("PolicyStore is required for training components")

        # Create losses tracker
        losses = Losses()

        # Create timer
        timer = Stopwatch(logger)
        timer.start()

        return cls(
            vecenv=vecenv,
            policy=policy,
            experience=experience,
            optimizer=optimizer,
            losses=losses,
            kickstarter=kickstarter,
            timer=timer,
            trainer_config=trainer_config,
            device=device_obj,
        )

    def rollout_step(self) -> Tuple[int, List[Any]]:
        """Perform a single rollout step.

        Returns:
            Tuple of (num_steps, info_list)
        """
        return perform_rollout_step(self.policy, self.vecenv, self.experience, self.device, self.timer)

    def is_ready_for_training(self) -> bool:
        """Check if experience buffer is ready for training."""
        return self.experience.ready_for_training

    def reset_for_rollout(self):
        """Reset experience buffer for new rollout."""
        self.experience.reset_for_rollout()

    def accumulate_stats(self, raw_infos: List[Any]):
        """Accumulate rollout statistics."""
        accumulate_rollout_stats(raw_infos, self.stats)

    def compute_advantages(self) -> torch.Tensor:
        """Compute advantages using GAE."""
        advantages = torch.zeros(self.experience.values.shape, device=self.device)
        initial_importance_sampling_ratio = torch.ones_like(self.experience.values)

        return compute_advantage(
            self.experience.values,
            self.experience.rewards,
            self.experience.dones,
            initial_importance_sampling_ratio,
            advantages,
            self.trainer_config.ppo.gamma,
            self.trainer_config.ppo.gae_lambda,
            self.trainer_config.vtrace.vtrace_rho_clip,
            self.trainer_config.vtrace.vtrace_c_clip,
            self.device,
        )

    def train_minibatch(
        self,
        minibatch: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Train on a single minibatch.

        Args:
            minibatch: Minibatch data
            advantages: Computed advantages

        Returns:
            Loss tensor
        """
        return process_minibatch_update(
            policy=self.policy,
            experience=self.experience,
            minibatch=minibatch,
            advantages=advantages,
            trainer_cfg=self.trainer_config,
            kickstarter=self.kickstarter,
            agent_step=self.agent_step,
            losses=self.losses,
            device=self.device,
        )

    def sample_minibatch(
        self,
        advantages: torch.Tensor,
        minibatch_idx: int,
        total_minibatches: int,
        anneal_beta: float,
    ) -> Dict[str, torch.Tensor]:
        """Sample a minibatch from experience buffer.

        Args:
            advantages: Computed advantages
            minibatch_idx: Current minibatch index
            total_minibatches: Total number of minibatches
            anneal_beta: Annealed beta for prioritized replay

        Returns:
            Minibatch dictionary
        """
        prio_cfg = self.trainer_config.prioritized_experience_replay
        return self.experience.sample_minibatch(
            advantages=advantages,
            prio_alpha=prio_cfg.prio_alpha,
            prio_beta=anneal_beta,
            minibatch_idx=minibatch_idx,
            total_minibatches=total_minibatches,
        )

    def optimize_step(self, loss: torch.Tensor, accumulate_steps: int):
        """Perform optimizer step with gradient accumulation.

        Args:
            loss: Loss tensor
            accumulate_steps: Number of steps to accumulate gradients
        """
        self.optimizer.zero_grad()
        loss.backward()

        if (self.epoch + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.trainer_config.ppo.max_grad_norm)
            self.optimizer.step()

            # Optional weight clipping
            if hasattr(self.policy, "clip_weights"):
                self.policy.clip_weights()

    def reset_training_state(self):
        """Reset state for new training epoch."""
        self.losses.zero()
        self.experience.reset_importance_sampling_ratios()


# Helper functions for common operations
def create_default_trainer_config(
    num_workers: int = 1,
    total_timesteps: int = 10_000_000,
    batch_size: int = 8192,
    minibatch_size: int = 512,
    checkpoint_dir: Optional[str] = None,
    **kwargs,
) -> TrainerConfig:
    """Create a default TrainerConfig with sensible values.

    Args:
        num_workers: Number of parallel workers
        total_timesteps: Total training timesteps
        batch_size: Batch size
        minibatch_size: Minibatch size
        checkpoint_dir: Directory for checkpoints. If not provided, uses
                       $DATA_DIR/$METTA_RUN/checkpoints or ./checkpoints
        **kwargs: Additional config overrides

    Returns:
        TrainerConfig instance
    """
    # Use environment variables for default paths if not provided
    if checkpoint_dir is None:
        run_name = os.environ.get("METTA_RUN", "default_run")
        data_dir = os.environ.get("DATA_DIR", "./train_dir")
        checkpoint_dir = os.path.join(data_dir, run_name, "checkpoints")

    config_dict = {
        "num_workers": num_workers,
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "checkpoint": {
            "checkpoint_dir": checkpoint_dir,
        },
        "simulation": {
            "replay_dir": "./replays",
        },
        "curriculum": "/env/mettagrid/simple",
    }

    # Apply overrides
    config_dict.update(kwargs)

    return TrainerConfig.model_validate(config_dict)


def save_checkpoint(
    policy: MettaAgent,
    policy_store: PolicyStore,
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> PolicyRecord:
    """Save a policy checkpoint.

    Args:
        policy: Policy to save
        policy_store: Policy store to use
        epoch: Current epoch
        metadata: Optional metadata

    Returns:
        Saved PolicyRecord
    """
    name = policy_store.make_model_name(epoch)
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = metadata or {}
    policy_record.policy = policy

    return policy_store.save(policy_record)


def load_checkpoint(
    policy_store: PolicyStore,
    path: str,
) -> PolicyRecord:
    """Load a policy checkpoint.

    Args:
        policy_store: Policy store to use
        path: Path to checkpoint

    Returns:
        Loaded PolicyRecord
    """
    return policy_store.policy_record(path)


def evaluate_policy(
    policy: MettaAgent,
    env_config: str,
    num_episodes: int = 10,
    device: str = "cuda",
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate a policy on an environment.

    Args:
        policy: Policy to evaluate
        env_config: Environment configuration path
        num_episodes: Number of episodes
        device: Device to use
        render: Whether to render

    Returns:
        Evaluation statistics
    """
    # Create evaluation environment - Environment returns a vecenv
    vecenv = Environment(
        curriculum_path=env_config,
        device=device,
        num_envs=1,
        is_training=False,
    )

    total_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs = vecenv.reset()  # type: ignore
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            with torch.no_grad():
                # Note: Real implementation would handle LSTM states properly
                action = policy(obs)

            obs, reward, terminated, truncated, info = vecenv.step(action)  # type: ignore
            done = terminated or truncated
            episode_reward += reward.sum()
            episode_length += 1

            if render:
                vecenv.render()  # type: ignore

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    vecenv.close()  # type: ignore

    return {
        "mean_reward": float(torch.tensor(total_rewards).mean()),
        "std_reward": float(torch.tensor(total_rewards).std()),
        "mean_length": float(torch.tensor(episode_lengths).mean()),
        "std_length": float(torch.tensor(episode_lengths).std()),
    }


def calculate_anneal_beta(
    epoch: int,
    total_timesteps: int,
    batch_size: int,
    prio_alpha: float,
    prio_beta0: float,
) -> float:
    """Calculate annealed beta for prioritized experience replay.

    Args:
        epoch: Current epoch
        total_timesteps: Total training timesteps
        batch_size: Batch size
        prio_alpha: Priority alpha
        prio_beta0: Initial beta value

    Returns:
        Annealed beta value
    """
    total_epochs = max(1, total_timesteps // batch_size)
    anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs
    return anneal_beta


# Re-export key classes for convenience
__all__ = [
    # Factory classes
    "Environment",
    "Agent",
    "TrainingComponents",
    # Config classes (from trainer_config)
    "TrainerConfig",
    "OptimizerConfig",
    "PPOConfig",
    "CheckpointConfig",
    "SimulationConfig",
    "PrioritizedExperienceReplayConfig",
    "VTraceConfig",
    "KickstartConfig",
    "TorchProfilerConfig",
    # Helper functions
    "create_default_trainer_config",
    "save_checkpoint",
    "load_checkpoint",
    "evaluate_policy",
    "calculate_anneal_beta",
    # Constants
    "TYPE_AGENT",
    "TYPE_WALL",
    "TYPE_MINE_RED",
    "TYPE_MINE_BLUE",
    "TYPE_MINE_GREEN",
    "TYPE_GENERATOR_RED",
    "TYPE_GENERATOR_BLUE",
    "TYPE_GENERATOR_GREEN",
    "TYPE_ALTAR",
    "TYPE_ARMORY",
    "TYPE_LASERY",
    "TYPE_LAB",
    "TYPE_FACTORY",
    "TYPE_TEMPLE",
    "TYPE_GENERIC_CONVERTER",
]
