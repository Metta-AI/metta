"""Test helpers for converting between old and new environment configurations."""

from typing import Optional

from omegaconf import OmegaConf

from metta.mettagrid.config import EnvConfig, PyGameConfig
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import LevelMap


def create_env_config_from_curriculum(curriculum: Curriculum, level: Optional[LevelMap] = None) -> EnvConfig:
    """Create EnvConfig from a curriculum for testing.

    This helper allows tests to continue using curriculum-based setup
    while the environments now use EnvConfig directly.
    """
    task = curriculum.get_task()
    task_cfg = task.env_cfg()

    # Extract game config
    if hasattr(task_cfg, "game"):
        # Already structured config
        game_dict = OmegaConf.to_container(task_cfg.game) if hasattr(task_cfg.game, "__dict__") else task_cfg.game
    else:
        # Assume task_cfg is the game config
        game_dict = OmegaConf.to_container(task_cfg) if hasattr(task_cfg, "__dict__") else task_cfg

    assert isinstance(game_dict, dict), "Game config must be a dictionary"

    # Create PyGameConfig from dict
    game_config = PyGameConfig(**game_dict)

    # Get level_map from task or build it
    if level is not None:
        # Use provided level
        level_map = level
    elif hasattr(game_config, "map_builder") and game_config.map_builder is not None:
        level_map = game_config.map_builder.build()
    else:
        # Create a minimal level map for testing
        import numpy as np

        grid = np.array([["empty"]], dtype="<U50")
        level_map = LevelMap(grid=grid, labels=[])

    return EnvConfig(game=game_config, level_map=level_map)
