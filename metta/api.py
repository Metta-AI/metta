"""
Clean API for Metta - provides direct instantiation without Hydra.

This API exposes the core training components from Metta, allowing users to:
1. Create environments, agents, and training components without Hydra
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.metta_agent import MettaAgent
from metta.common.profiling.stopwatch import Stopwatch
from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum, Task
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functions import (
    accumulate_rollout_stats,
    compute_advantage,
    perform_rollout_step,
    process_minibatch_update,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import (
    CheckpointConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    TrainerConfig,
)
from metta.rl.vecenv import make_vecenv

logger = logging.getLogger(__name__)


# Named tuple for run directories
class RunDirectories:
    """Container for run directory paths."""

    def __init__(self, run_dir: str, checkpoint_dir: str, replay_dir: str, stats_dir: str, run_name: str):
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir
        self.replay_dir = replay_dir
        self.stats_dir = stats_dir
        self.run_name = run_name


def setup_run_directories(run_name: Optional[str] = None, data_dir: Optional[str] = None) -> RunDirectories:
    """Set up the directory structure for a training run.

    This creates the same directory structure as tools/train.py:
    - {data_dir}/{run_name}/
        - checkpoints/  # Model checkpoints
        - replays/      # Replay files
        - stats/        # Evaluation statistics

    Args:
        run_name: Name for this run. If not provided, uses METTA_RUN env var or timestamp
        data_dir: Base data directory. If not provided, uses DATA_DIR env var or ./train_dir

    Returns:
        RunDirectories object with all directory paths
    """
    # Get run name and data directory
    if run_name is None:
        run_name = os.environ.get("METTA_RUN", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./train_dir")

    # Create paths
    run_dir = os.path.join(data_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    replay_dir = os.path.join(run_dir, "replays")
    stats_dir = os.path.join(run_dir, "stats")

    # Create directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(replay_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Run name: {run_name}")

    return RunDirectories(
        run_dir=run_dir, checkpoint_dir=checkpoint_dir, replay_dir=replay_dir, stats_dir=stats_dir, run_name=run_name
    )


def save_experiment_config(
    dirs: RunDirectories,
    device: torch.device,
    trainer_config: TrainerConfig,
) -> None:
    """Save training configuration to config.yaml in the run directory.

    This builds the experiment configuration from the provided components
    and saves it for reproducibility.

    Args:
        dirs: RunDirectories object with paths
        device: Training device
        trainer_config: TrainerConfig object with training parameters
    """
    from omegaconf import OmegaConf

    # Build experiment configuration
    experiment_config = {
        "run": dirs.run_name,
        "run_dir": dirs.run_dir,
        "data_dir": os.path.dirname(dirs.run_dir),
        "device": str(device),
        "trainer": {
            "num_workers": trainer_config.num_workers,
            "total_timesteps": trainer_config.total_timesteps,
            "batch_size": trainer_config.batch_size,
            "minibatch_size": trainer_config.minibatch_size,
            "checkpoint_dir": dirs.checkpoint_dir,
            "optimizer": trainer_config.optimizer.model_dump(),
            "ppo": trainer_config.ppo.model_dump(),
            "checkpoint": trainer_config.checkpoint.model_dump(),
            "simulation": trainer_config.simulation.model_dump(),
            "profiler": trainer_config.profiler.model_dump(),
        },
    }

    # Save to file
    config_path = os.path.join(dirs.run_dir, "config.yaml")
    OmegaConf.save(experiment_config, config_path)
    logger.info(f"Saved config to {config_path}")


# Helper to create default environment config
def _get_default_env_config(num_agents: int = 4, width: int = 32, height: int = 32) -> Dict[str, Any]:
    """Get default environment configuration for navigation training."""
    # Object type IDs from mettagrid/src/metta/mettagrid/objects/constants.hpp
    TYPE_MINE_RED = 2
    TYPE_GENERATOR_RED = 5
    TYPE_ALTAR = 8
    TYPE_WALL = 1
    TYPE_BLOCK = 14

    return {
        "sampling": 1,  # Enable sampling for navigation
        "game": {
            "max_steps": 1000,
            "num_agents": num_agents,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "inventory_item_names": [
                "ore_red",
                "ore_blue",
                "ore_green",
                "battery_red",
                "battery_blue",
                "battery_green",
                "heart",
                "armor",
                "laser",
                "blueprint",
            ],
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
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
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
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 1000,
                    "initial_items": 1,
                    "color": 1,
                },
                "wall": {"type_id": TYPE_WALL, "swappable": False},
                "block": {"type_id": TYPE_BLOCK, "swappable": True},
            },
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.multi_room.MultiRoom",
                "num_rooms": num_agents,
                "border_width": 6,
                "room": {
                    "_target_": "metta.mettagrid.room.terrain_from_numpy.TerrainFromNumpy",
                    "border_width": 3,
                    "agents": 1,
                    "dir": "terrain_maps_nohearts",  # Default terrain directory
                    "objects": {
                        "altar": 30,  # Default altar count
                    },
                },
            },
        },
    }


def _get_default_agent_config(device: str = "cuda") -> Dict[str, Any]:
    """Get default agent configuration based on fast.yaml architecture."""
    return {
        "device": device,
        "agent": {
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "l2_init_weight_update_interval": 0,
            "observations": {"obs_key": "grid_obs"},
            "components": {
                "_obs_": {
                    "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                    "sources": None,
                },
                "obs_normalizer": {
                    "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                    "sources": [{"name": "_obs_"}],
                },
                "cnn1": {
                    "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                    "sources": [{"name": "obs_normalizer"}],
                    "nn_params": {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                },
                "cnn2": {
                    "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                    "sources": [{"name": "cnn1"}],
                    "nn_params": {
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                },
                "obs_flattener": {
                    "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                    "sources": [{"name": "cnn2"}],
                },
                "encoded_obs": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "obs_flattener"}],
                    "nn_params": {"out_features": 512},
                },
                "_core_": {
                    "_target_": "metta.agent.lib.lstm.LSTM",
                    "sources": [{"name": "encoded_obs"}],
                    "output_size": 512,
                    "nn_params": {
                        "num_layers": 1,
                    },
                },
                "_value_": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 1},
                    "nonlinearity": None,
                },
                "actor_1": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 512},
                },
                "_action_embeds_": {
                    "_target_": "metta.agent.lib.action.ActionEmbedding",
                    "sources": None,
                    "nn_params": {
                        "num_embeddings": 100,
                        "embedding_dim": 16,
                    },
                },
                "_action_": {
                    "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                    "sources": [
                        {"name": "actor_1"},
                        {"name": "_action_embeds_"},
                    ],
                },
            },
        },
    }


# Helper class for navigation curriculum without Hydra
class NavigationBucketedCurriculum(Curriculum):
    """Navigation curriculum that cycles through different terrain types without using Hydra."""

    def __init__(self, base_config: Dict[str, Any], terrain_dirs: List[str], altar_range: Tuple[int, int]):
        self.base_config = DictConfig(base_config)
        self.terrain_dirs = terrain_dirs
        self.altar_range = altar_range
        self.current_idx = 0

    def get_task(self) -> "Task":
        import random

        from metta.mettagrid.curriculum.core import Task

        # Select a random terrain
        terrain_dir = random.choice(self.terrain_dirs)

        # Select a random altar count
        altar_count = random.randint(self.altar_range[0], self.altar_range[1])

        # Create task config
        task_config = OmegaConf.create(self.base_config)
        OmegaConf.set_struct(task_config, False)

        # Update the terrain directory
        task_config.game.map_builder.room.dir = terrain_dir

        # Update the altar count
        task_config.game.map_builder.room.objects.altar = altar_count

        # Create task name
        task_name = f"terrain={terrain_dir};altar={altar_count}"

        return Task(task_name, self, task_config)

    def get_task_probs(self) -> dict[str, float]:
        """Return uniform probabilities for all terrain types."""
        prob = 1.0 / len(self.terrain_dirs)
        return {terrain: prob for terrain in self.terrain_dirs}


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
            curriculum_path: Optional path to environment or curriculum configuration
                           (e.g., "/env/mettagrid/simple" or "/env/mettagrid/curriculum/navigation/bucketed")
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
        logger.info("Creating environment...")

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
                    if "num_rooms" in env_config["game"]["map_builder"]:
                        env_config["game"]["map_builder"]["num_rooms"] = num_agents
            if width is not None:
                if "map_builder" in env_config["game"]:
                    if "room" in env_config["game"]["map_builder"]:
                        env_config["game"]["map_builder"]["room"]["width"] = width
                    else:
                        env_config["game"]["map_builder"]["width"] = width
            if height is not None:
                if "map_builder" in env_config["game"]:
                    if "room" in env_config["game"]["map_builder"]:
                        env_config["game"]["map_builder"]["room"]["height"] = height
                    else:
                        env_config["game"]["map_builder"]["height"] = height

        # Create curriculum
        if curriculum_path == "/env/mettagrid/curriculum/navigation/bucketed":
            # Special handling for bucketed navigation curriculum
            terrain_dirs = [
                "terrain_maps_nohearts",
                "varied_terrain/balanced_large",
                "varied_terrain/balanced_medium",
                "varied_terrain/balanced_small",
                "varied_terrain/sparse_large",
                "varied_terrain/sparse_medium",
                "varied_terrain/sparse_small",
                "varied_terrain/dense_large",
                "varied_terrain/dense_medium",
                "varied_terrain/dense_small",
                "varied_terrain/maze_large",
                "varied_terrain/maze_medium",
                "varied_terrain/maze_small",
                "varied_terrain/cylinder-world_large",
                "varied_terrain/cylinder-world_medium",
                "varied_terrain/cylinder-world_small",
            ]

            # Create navigation training template config
            template_config = _get_default_env_config(
                num_agents=num_agents or 4, width=width or 32, height=height or 32
            )

            # Ensure sampling is disabled for evaluation
            template_config["sampling"] = 0

            # Create the custom navigation curriculum
            curriculum = NavigationBucketedCurriculum(
                base_config=template_config,
                terrain_dirs=terrain_dirs,
                altar_range=(10, 50),
            )
        elif curriculum_path:
            # For other curriculum paths, try to create a simple single-task curriculum
            # by using the path as a task name with the provided config
            task_config = DictConfig(env_config)
            curriculum = SingleTaskCurriculum(curriculum_path, task_config)
        else:
            # Create a single task curriculum with the provided config
            task_config = DictConfig(env_config)
            curriculum_name = "custom_env"
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
        config: Optional[DictConfig] = None,
        device: str = "cuda",
    ) -> MettaAgent:
        """Create a Metta agent.

        Args:
            env: Vectorized environment (from Environment factory)
            config: Optional DictConfig with agent configuration. If not provided, uses a default configuration.
            device: Device to use

        Returns:
            MettaAgent instance
        """
        logger.info("Creating agent...")

        # Use default config if none provided
        if config is None:
            config = DictConfig(_get_default_agent_config(device))

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


class Optimizer:
    """Wrapper for PyTorch optimizers with gradient accumulation and clipping support.

    This provides a clean interface for the optimization step, handling:
    - Gradient accumulation across minibatches
    - Gradient clipping
    - Optional weight clipping on the policy
    """

    def __init__(
        self,
        optimizer_type: str = "adam",
        policy: Optional[MettaAgent] = None,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
    ):
        """Initialize optimizer wrapper.

        Args:
            optimizer_type: Type of optimizer ("adam" or "muon")
            policy: Policy to optimize
            learning_rate: Learning rate
            betas: Beta parameters for Adam/Muon
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        if policy is None:
            raise ValueError("Policy must be provided to Optimizer")
        logger.info(f"Creating optimizer... Using {optimizer_type.capitalize()} optimizer with lr={learning_rate}")

        self.policy = policy
        self.max_grad_norm = max_grad_norm

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=float(weight_decay),  # type: ignore - PyTorch accepts float
            )
        elif optimizer_type == "muon":
            from heavyball import ForeachMuon

            self.optimizer = ForeachMuon(
                policy.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=float(weight_decay),  # type: ignore - PyTorch accepts float
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adam' or 'muon'")

    def step(self, loss: torch.Tensor, epoch: int, accumulate_steps: int = 1):
        """Perform optimization step with gradient accumulation.

        Args:
            loss: Loss tensor to backpropagate
            epoch: Current epoch (for accumulation check)
            accumulate_steps: Number of steps to accumulate gradients
        """
        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Optional weight clipping
            if hasattr(self.policy, "clip_weights"):
                self.policy.clip_weights()

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access to optimizer param groups (for learning rate etc)."""
        return self.optimizer.param_groups


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
    # New wrapper classes
    "Optimizer",
    # Training components (for direct instantiation)
    "Experience",
    "Kickstarter",
    "Losses",
    "Stopwatch",
    # Config classes (from trainer_config)
    "TrainerConfig",
    "OptimizerConfig",
    "PPOConfig",
    "CheckpointConfig",
    "SimulationConfig",
    # Helper functions
    "calculate_anneal_beta",
    "setup_run_directories",
    "save_experiment_config",
    # Functions from rl.functions (commonly used)
    "perform_rollout_step",
    "process_minibatch_update",
    "accumulate_rollout_stats",
    "compute_advantage",  # Export the real function directly
    # Helper classes
    "RunDirectories",
]
