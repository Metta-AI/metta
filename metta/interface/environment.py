"""Environment factory and helpers for Metta."""

import logging
import random
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum, Task
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.vecenv import make_vecenv

logger = logging.getLogger(__name__)


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

    def get_task_probs(self) -> Dict[str, float]:
        """Return the current task probability for logging purposes."""
        task_name = f"prebuilt({self._env_name})"
        return {task_name: 1.0}


class NavigationBucketedCurriculum(Curriculum):
    """Navigation curriculum that cycles through different terrain types without using Hydra."""

    def __init__(self, base_config: Dict[str, Any], terrain_dirs: list[str], altar_range: tuple[int, int]):
        self.base_config = DictConfig(base_config)
        self.terrain_dirs = terrain_dirs
        self.altar_range = altar_range
        self.current_idx = 0

    def get_task(self) -> Task:
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

    def get_task_probs(self) -> Dict[str, float]:
        """Return uniform probabilities for all terrain types."""
        prob = 1.0 / len(self.terrain_dirs)
        return {terrain: prob for terrain in self.terrain_dirs}


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
                "change_color": {"enabled": True},
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
            template_config = env_config or _get_default_env_config(
                num_agents=num_agents or 512, width=width or 32, height=height or 32
            )

            # Create the custom navigation curriculum
            curriculum = NavigationBucketedCurriculum(
                base_config=template_config,
                terrain_dirs=terrain_dirs,
                altar_range=(2, 18),  # Based on bucketed.yaml config
            )
        elif curriculum_path:
            # Use the existing curriculum loading system
            env_overrides = DictConfig({})
            if env_config and curriculum_path.startswith("/env/mettagrid/curriculum/"):
                # Apply env_config as overrides for curriculum
                env_overrides = DictConfig(env_config)
            curriculum = curriculum_from_config_path(curriculum_path, env_overrides)
        else:
            # Create a single task curriculum with the provided or default config
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
