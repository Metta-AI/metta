"""
Clean API for Metta - provides direct instantiation without Hydra.

This API exposes the core training components from Metta, allowing users to:
1. Create environments, agents, and training components programmatically
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
4. Support distributed training with minimal setup
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
from metta.common.util.fs import wait_for_file
from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum, Task
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.functions import (
    cleanup_old_policies,
)
from metta.rl.trainer_config import (
    TrainerConfig,
)
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Directory Management (used first in run.py)
# ============================================================================


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


# ============================================================================
# Distributed Training Setup
# ============================================================================


def setup_distributed_training(base_device: str = "cuda") -> Tuple[torch.device, bool, int, int]:
    """Set up device and distributed training, returning all needed information.

    This combines device setup and distributed initialization into a single call,
    matching the initialization pattern from tools/train.py.

    Args:
        base_device: Base device string ("cuda" or "cpu")

    Returns:
        Tuple of (device, is_master, world_size, rank)
        - device: The torch.device to use for training
        - is_master: True if this is the master process (rank 0)
        - world_size: Total number of processes (1 if not distributed)
        - rank: Current process rank (0 if not distributed)
    """
    # Check if we're in a distributed environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        # For CUDA, use device with local rank
        if base_device.startswith("cuda"):
            device = torch.device(f"{base_device}:{local_rank}")
            backend = "nccl"
        else:
            # For CPU, just use cpu device (no rank suffix)
            device = torch.device(base_device)
            backend = "gloo"

        torch.distributed.init_process_group(backend=backend)

        # Get distributed info after initialization
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_master = rank == 0
    else:
        # Single GPU or CPU
        device = torch.device(base_device)
        rank = 0
        world_size = 1
        is_master = True

    logger.info(f"Using device: {device} (rank {rank}/{world_size})")
    return device, is_master, world_size, rank


def cleanup_distributed():
    """Clean up distributed training if it was initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_wandb(
    run_name: str,
    run_dir: str,
    enabled: bool = True,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    job_type: str = "train",
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any]]:  # Returns (wandb_run, wandb_ctx)
    """Initialize Weights & Biases logging with proper configuration.

    This helper function creates the wandb configuration in the format expected
    by WandbContext, handling both Hydra and non-Hydra use cases. It generates
    the same configuration structure that the Hydra pipeline creates via
    configs/wandb/*.yaml files.

    Args:
        run_name: Name of the run (used for group, name, and run_id)
        run_dir: Directory where run data is stored
        enabled: Whether wandb logging is enabled
        project: W&B project name (defaults to env var or "metta")
        entity: W&B entity name (defaults to env var or "metta-research")
        config: Optional configuration dict to log
        job_type: Type of job (e.g., "train", "eval")
        tags: Optional list of tags
        notes: Optional notes for the run

    Returns:
        Tuple of (wandb_run, wandb_ctx):
        - wandb_run: The W&B run object if initialized, None otherwise
        - wandb_ctx: The WandbContext object for cleanup

    Example:
        >>> wandb_run, wandb_ctx = initialize_wandb(
        ...     run_name=dirs.run_name,
        ...     run_dir=dirs.run_dir,
        ...     enabled=not os.environ.get("WANDB_DISABLED"),
        ...     config={"trainer": trainer_config.model_dump()}
        ... )

    Note:
        This function is compatible with the Hydra pipeline used in tools/train.py.
        It creates the same wandb configuration structure that would be loaded from
        configs/wandb/metta_research.yaml or configs/wandb/off.yaml.
    """
    from metta.common.wandb.wandb_context import WandbContext

    # Build wandb config
    if enabled:
        wandb_config = {
            "enabled": True,
            "project": project or os.environ.get("WANDB_PROJECT", "metta"),
            "entity": entity or os.environ.get("WANDB_ENTITY", "metta-research"),
            "group": run_name,
            "name": run_name,
            "run_id": run_name,
            "data_dir": run_dir,
            "job_type": job_type,
            "tags": tags or [],
            "notes": notes or "",
        }
    else:
        wandb_config = {"enabled": False}

    # Build global config for WandbContext
    # This mimics what Hydra would provide
    global_config = {
        "run": run_name,
        "run_dir": run_dir,
        "cmd": job_type,
        "wandb": wandb_config,
    }

    # Add any user-provided config
    if config:
        global_config.update(config)

    # Initialize wandb context
    wandb_ctx = WandbContext(DictConfig(wandb_config), DictConfig(global_config))
    wandb_run = wandb_ctx.__enter__()

    return wandb_run, wandb_ctx


def cleanup_wandb(wandb_ctx: Optional[Any]) -> None:
    """Clean up wandb context if it exists.

    Args:
        wandb_ctx: The WandbContext object returned by initialize_wandb
    """
    if wandb_ctx is not None:
        wandb_ctx.__exit__(None, None, None)


# ============================================================================
# Configuration Management
# ============================================================================


def save_experiment_config(
    dirs: RunDirectories,
    device: torch.device,
    trainer_config: TrainerConfig,
) -> None:
    """Save training configuration to config.yaml in the run directory.

    This builds the experiment configuration from the provided components
    and saves it for reproducibility. Only saves on master rank in distributed mode.

    Args:
        dirs: RunDirectories object with paths
        device: Training device
        trainer_config: TrainerConfig object with training parameters
    """
    # Only save on master rank to avoid file conflicts
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

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


# ============================================================================
# Helper Classes and Functions (used internally by Environment and Agent)
# ============================================================================


class PreBuiltConfigCurriculum(Curriculum):
    """A curriculum that uses a pre-built config instead of loading from Hydra.

    This allows us to bypass Hydra entirely when running evaluation or replay
    generation without having Hydra initialized.
    """

    def __init__(self, env_name: str, pre_built_config: Any):
        self._env_name = env_name
        self._cfg_template = pre_built_config

    def get_task(self) -> Task:
        """Return a task with the pre-built config."""
        return Task(f"prebuilt({self._env_name})", self, self._cfg_template)


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
                "default_resource_limit": 50,
                "resource_limits": {
                    "heart": 255,
                },
                "freeze_duration": 10,
                "rewards": {
                    "inventory": {
                        "ore_red": 0.01,
                        "battery_red": 0.02,
                        "heart": 1,
                        "ore_red_max": 10,
                        "battery_red_max": 10,
                    }
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
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "objects": {
                "mine_red": {
                    "type_id": TYPE_MINE_RED,
                    "output_resources": {"ore_red": 1},
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 0,
                    "initial_resource_count": 0,
                    "color": 0,
                },
                "generator_red": {
                    "type_id": TYPE_GENERATOR_RED,
                    "input_resources": {"ore_red": 1},
                    "output_resources": {"battery_red": 1},
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 0,
                    "initial_resource_count": 0,
                    "color": 0,
                },
                "altar": {
                    "type_id": TYPE_ALTAR,
                    "input_resources": {"battery_red": 3},
                    "output_resources": {"heart": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 1000,
                    "initial_resource_count": 1,
                    "color": 1,
                },
                "wall": {"type_id": TYPE_WALL, "swappable": False},
                "block": {"type_id": TYPE_BLOCK, "swappable": True},
            },
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


def _get_default_agent_config(device: str = "cuda") -> DictConfig:
    """Get default agent configuration based on fast.yaml architecture."""
    return DictConfig(
        {
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
    )


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


# ============================================================================
# Environment Factory
# ============================================================================


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
                           (e.g., "/env/mettagrid/arena/advanced" or "/env/mettagrid/curriculum/navigation/bucketed")
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


# ============================================================================
# Agent Factory
# ============================================================================


class Agent:
    """Factory for creating Metta agents with a clean API.

    This handles agent creation and initialization without Hydra.
    """

    def __new__(
        cls,
        env: Any,  # vecenv wrapper
        config: Optional[DictConfig] = None,
        device: str = "cuda",
    ) -> Any:  # Returns MettaAgent or DistributedMettaAgent
        """Create a Metta agent.

        Args:
            env: Vectorized environment (from Environment factory)
            config: Optional DictConfig with agent configuration. If not provided, uses a default configuration.
            device: Device to use

        Returns:
            MettaAgent instance (or DistributedMettaAgent wrapper if distributed is initialized)
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


# ============================================================================
# Checkpoint Management
# ============================================================================


def load_checkpoint(
    checkpoint_dir: str,
    agent: Any,
    optimizer: Optional[Any] = None,
    policy_store: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, Optional[str]]:
    """Load training checkpoint if it exists.

    This is a simplified version that just loads the checkpoint without
    handling distributed coordination or initial policy creation.

    Args:
        checkpoint_dir: Directory containing checkpoints
        agent: The agent/policy to potentially update
        optimizer: Optional optimizer to restore state to
        policy_store: Optional PolicyStore (not used in simplified version)
        device: Device (not used in simplified version)

    Returns:
        Tuple of (agent_step, epoch, policy_path)
        - agent_step: Current training step (0 if no checkpoint)
        - epoch: Current epoch (0 if no checkpoint)
        - policy_path: Path to loaded policy (None if no checkpoint)
    """
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    # Try to load existing checkpoint
    existing_checkpoint = TrainerCheckpoint.load(checkpoint_dir)

    if existing_checkpoint:
        # Restore training state
        agent_step = existing_checkpoint.agent_step
        epoch = existing_checkpoint.epoch

        # Load policy state if agent provided and policy path exists
        if agent is not None and existing_checkpoint.policy_path and policy_store is not None:
            try:
                policy_pr = policy_store.policy_record(existing_checkpoint.policy_path)
                agent.load_state_dict(policy_pr.policy.state_dict())
            except Exception as e:
                logger.warning(f"Failed to load policy state: {e}")

        # Load optimizer state if provided
        if optimizer is not None and existing_checkpoint.optimizer_state_dict:
            try:
                if hasattr(optimizer, "optimizer"):
                    # Handle our Optimizer wrapper
                    optimizer.optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
                elif hasattr(optimizer, "load_state_dict"):
                    # Handle raw PyTorch optimizer
                    optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
            except ValueError:
                logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

        return agent_step, epoch, existing_checkpoint.policy_path

    # No checkpoint found
    return 0, 0, None


def save_checkpoint(
    epoch: int,
    agent_step: int,
    agent: Any,
    optimizer: Any,
    policy_store: Any,
    checkpoint_path: str,
    checkpoint_interval: int,
    stats: Optional[Dict[str, Any]] = None,
    force_save: bool = False,
) -> Optional[Any]:
    """Save a training checkpoint including policy and training state.

    In distributed mode, only the master process saves. Callers are responsible
    for adding barriers if synchronization is needed.

    Args:
        epoch: Current training epoch
        agent_step: Current agent step count
        agent: The agent/policy to save
        optimizer: The optimizer with state to save
        policy_store: PolicyStore instance for saving policies
        checkpoint_path: Directory path for saving checkpoints
        checkpoint_interval: How often to save checkpoints
        stats: Optional statistics dictionary to include in metadata
        force_save: Force save even if not on checkpoint interval

    Returns:
        The saved policy record if saved, None otherwise
    """
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    should_save = force_save or (epoch % checkpoint_interval == 0)
    if not should_save:
        return None

    # Only master saves in distributed mode
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    # Master (or single GPU) saves the checkpoint
    logger.info(f"Saving checkpoint at epoch {epoch}")

    # Extract the actual policy module from distributed wrapper if needed
    from torch.nn.parallel import DistributedDataParallel

    from metta.agent.metta_agent import DistributedMettaAgent

    policy_to_save = agent
    if isinstance(agent, DistributedMettaAgent):
        policy_to_save = agent.module
    elif isinstance(agent, DistributedDataParallel):
        policy_to_save = agent.module

    # Create policy record directly
    name = policy_store.make_model_name(epoch)
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = {
        "agent_step": agent_step,
        "epoch": epoch,
        "stats": dict(stats) if stats else {},
        "final": force_save,  # Mark if this is the final checkpoint
    }
    policy_record.policy = policy_to_save

    # Save through policy store
    saved_policy_record = policy_store.save(policy_record)

    # Save training state
    # Get optimizer state dict
    optimizer_state_dict = None
    if optimizer is not None:
        if hasattr(optimizer, "optimizer"):
            # Handle our Optimizer wrapper
            optimizer_state_dict = optimizer.optimizer.state_dict()
        elif hasattr(optimizer, "state_dict"):
            # Handle raw PyTorch optimizer
            optimizer_state_dict = optimizer.state_dict()

    trainer_checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer_state_dict=optimizer_state_dict,
        policy_path=saved_policy_record.uri if hasattr(saved_policy_record, "uri") else None,
        stopwatch_state=None,
    )
    trainer_checkpoint.save(checkpoint_path)

    # Clean up old policies to prevent disk space issues
    if epoch % 10 == 0:
        cleanup_old_policies(checkpoint_path, keep_last_n=5)

    return saved_policy_record


def wrap_agent_distributed(agent: Any, device: torch.device) -> Any:
    """Wrap agent in DistributedMettaAgent if distributed training is initialized.

    Args:
        agent: The agent to potentially wrap
        device: The device to use

    Returns:
        The agent, possibly wrapped in DistributedMettaAgent
    """
    if torch.distributed.is_initialized():
        from torch.nn.parallel import DistributedDataParallel

        from metta.agent.metta_agent import DistributedMettaAgent

        # For CPU, we need to handle DistributedDataParallel differently
        if device.type == "cpu":
            # Convert BatchNorm to SyncBatchNorm
            agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
            # For CPU, don't pass device_ids
            agent = DistributedDataParallel(agent)
        else:
            # For GPU, use the custom DistributedMettaAgent wrapper
            agent = DistributedMettaAgent(agent, device)

    return agent


def ensure_initial_policy(
    agent: Any,
    policy_store: Any,
    checkpoint_path: str,
    loaded_policy_path: Optional[str],
    device: torch.device,
) -> None:
    """Ensure all ranks have the same initial policy in distributed training.

    If no checkpoint exists, master creates and saves the initial policy,
    then all ranks synchronize. In single GPU mode, just saves the initial policy.

    Args:
        agent: The agent to initialize
        policy_store: PolicyStore instance
        checkpoint_path: Directory for checkpoints
        loaded_policy_path: Path to already loaded policy (None if no checkpoint)
        device: Training device
    """
    # If we already loaded a policy, nothing to do
    if loaded_policy_path is not None:
        return

    # Get distributed info
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        is_master = rank == 0
    else:
        rank = 0
        is_master = True

    if torch.distributed.is_initialized():
        if is_master:
            # Master creates and saves initial policy
            save_checkpoint(
                epoch=0,
                agent_step=0,
                agent=agent,
                optimizer=None,
                policy_store=policy_store,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=1,  # Force save
                stats={},
                force_save=True,
            )
            # Master waits at barrier after saving
            torch.distributed.barrier()
        else:
            # Non-master ranks wait at barrier first
            torch.distributed.barrier()

            # Then load the policy master created
            default_policy_path = os.path.join(checkpoint_path, policy_store.make_model_name(0))
            if not wait_for_file(default_policy_path, timeout=300):
                raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_policy_path}")

            # Load the policy
            policy_pr = policy_store.policy_record(default_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())  # type: ignore
    else:
        # Single GPU mode creates and saves initial policy
        save_checkpoint(
            epoch=0,
            agent_step=0,
            agent=agent,
            optimizer=None,
            policy_store=policy_store,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            stats={},
            force_save=True,
        )


# ============================================================================
# Optimizer Wrapper
# ============================================================================


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


# ============================================================================
# Training Loop Helper Functions
# ============================================================================


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


# ============================================================================
# Evaluation and Replay Configuration
# ============================================================================


def create_evaluation_config_suite() -> SimulationSuiteConfig:
    """Create evaluation suite configuration with pre-built configs to bypass Hydra.

    This creates the standard navigation evaluation suite used by the training system,
    but with pre-built configs that don't require Hydra to be initialized.

    Returns:
        SimulationSuiteConfig with navigation tasks
    """
    from metta.sim.simulation_config import SimulationSuiteConfig

    # Create pre-built navigation evaluation configs
    base_nav_config = _get_default_env_config()
    base_nav_config["sampling"] = 0  # Disable sampling for evaluation

    # Create evaluation configs for different terrain sizes
    simulations = {}

    # Small terrain evaluation
    simulations["navigation/terrain_small"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_small"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Medium terrain evaluation
    simulations["navigation/terrain_medium"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_medium"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Large terrain evaluation
    simulations["navigation/terrain_large"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_large"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Create suite config
    evaluation_config = SimulationSuiteConfig(
        name="evaluation",
        simulations=simulations,
        num_episodes=10,  # Will be overridden by individual configs
        env_overrides={},  # Suite-level overrides
    )

    return evaluation_config


def create_replay_config(terrain_dir: str = "varied_terrain/balanced_medium") -> SingleEnvSimulationConfig:
    """Create a pre-built replay configuration to bypass Hydra.

    Args:
        terrain_dir: Directory for terrain maps (default: varied_terrain/balanced_medium)

    Returns:
        SingleEnvSimulationConfig with pre-built config attached
    """
    from metta.sim.simulation_config import SingleEnvSimulationConfig

    # Create pre-built navigation config for replay
    replay_config = _get_default_env_config()
    replay_config["sampling"] = 0  # Disable sampling for replay

    # Create simulation config with pre-built config in env_overrides
    replay_sim_config = SingleEnvSimulationConfig(
        env="/env/mettagrid/navigation/training/terrain_from_numpy",
        num_episodes=1,
        max_time_s=60,
        env_overrides={
            "game": {"map_builder": {"room": {"dir": terrain_dir}}},
            "_pre_built_env_config": DictConfig(replay_config),
        },
    )

    return replay_sim_config


# ============================================================================
# Export List
# ============================================================================

# Re-export key classes for convenience
__all__ = [
    # Factory classes
    "Environment",
    "Agent",
    # New wrapper classes
    "Optimizer",
    # Helper functions unique to api.py
    "calculate_anneal_beta",
    "setup_run_directories",
    "save_experiment_config",
    "save_checkpoint",
    "setup_distributed_training",
    "cleanup_distributed",
    "initialize_wandb",
    "cleanup_wandb",
    "load_checkpoint",
    "wrap_agent_distributed",
    "ensure_initial_policy",
    # Helper classes
    "RunDirectories",
    "PreBuiltConfigCurriculum",
    # Evaluation/replay configuration
    "create_evaluation_config_suite",
    "create_replay_config",
]
