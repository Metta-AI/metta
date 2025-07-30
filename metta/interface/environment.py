"""Environment factory and helpers for Metta."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from metta.mettagrid.curriculum import (
    Curriculum,
    bucketed_task_set,
    single_task,
)
from metta.rl.vecenv import make_vecenv

logger = logging.getLogger(__name__)


def _get_default_env_config(num_agents: int = 4, width: int = 32, height: int = 32) -> Dict[str, Any]:
    """Get default environment configuration for navigation training."""
    # Object type IDs from mettagrid/src/metta/mettagrid/objects/constants.hpp
    TYPE_MINE_RED = 2
    TYPE_GENERATOR_RED = 5
    TYPE_ALTAR = 8
    TYPE_WALL = 1
    TYPE_BLOCK = 14

    return {
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


def navigation_bucketed_task_set(
    base_config: Dict[str, Any],
    terrain_dirs: List[str],
    altar_range: Tuple[int, int],
    name: str = "navigation_bucketed",
) -> Curriculum:  # Returns Curriculum
    """Create a navigation task set with terrain and altar variations.
    Args:
        base_config: Base environment configuration
        terrain_dirs: List of terrain directory names to sample from
        altar_range: Tuple of (min_altars, max_altars) for random sampling
        name: Name for the Curriculum root

    Returns:
        Curriculum with tasks for each terrain/altar combination
    """

    # Convert terrain_dirs to discrete values
    terrain_values = terrain_dirs

    # Convert altar_range to discrete values (sample a few representative values)
    min_altars, max_altars = altar_range
    # Create a reasonable sampling of the altar range
    if max_altars - min_altars <= 10:
        # Small range: include all values
        altar_values = list(range(min_altars, max_altars + 1))
    else:
        # Large range: sample representative values
        altar_values = [
            min_altars,
            min_altars + (max_altars - min_altars) // 4,
            min_altars + (max_altars - min_altars) // 2,
            min_altars + 3 * (max_altars - min_altars) // 4,
            max_altars,
        ]

    # Define parameter buckets for the grid
    buckets = {
        "game.map_builder.room.dir": {"values": terrain_values},
        "game.map_builder.room.objects.altar": {"values": altar_values},
    }

    # Create the task set using bucketed_task_set
    return bucketed_task_set(name=name, env_cfg_template=DictConfig(base_config), buckets=buckets)


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

            # Create the custom navigation curriculum using the new function
            curriculum = navigation_bucketed_task_set(
                base_config=template_config, terrain_dirs=terrain_dirs, altar_range=(10, 50), name="navigation_bucketed"
            )
        elif curriculum_path:
            # For other curriculum paths, try to create a simple single-task curriculum
            # by using the path as a task name with the provided config
            task_config = DictConfig(env_config)
            curriculum = single_task(curriculum_path, task_config)
        else:
            # Create a single task curriculum with the provided config
            task_config = DictConfig(env_config)
            curriculum_name = "custom_env"
            curriculum = single_task(curriculum_name, task_config)

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
