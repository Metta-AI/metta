"""
MettaGridPufferBase - Base PufferLib integration for MettaGrid.

This class provides the core PufferLib integration that is shared between
the user adapter (MettaGridPufferEnv) and training environment (MettaGridEnv).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
from pufferlib import PufferEnv
from typing_extensions import override

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)


class MettaGridPufferBase(MettaGridCore, PufferEnv):
    """
    Base class for PufferLib integration with MettaGrid.

    This class handles the common PufferLib integration logic that is shared
    between user adapters and training environments. It combines MettaGridCore
    with PufferEnv to provide PufferLib compatibility.

    Inherits from:
    - MettaGridCore: Core C++ environment wrapper functionality
    - pufferlib.PufferEnv: High-performance vectorized environment interface
      https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments.py
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        buf: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize PufferLib base environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            buf: PufferLib buffer object
            **kwargs: Additional arguments
        """
        # Store curriculum
        self._curriculum = curriculum
        self._task = curriculum.get_task()

        # Get level from curriculum if not provided
        if level is None:
            level = self._task.env_cfg().game.map_builder.build()

        # Ensure we have a level
        assert level is not None, "Level must be provided or generated from curriculum"

        # Get game config for core initialization
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Initialize core environment
        MettaGridCore.__init__(
            self,
            level=level,
            game_config_dict=game_config_dict,
            render_mode=render_mode,
            **kwargs,
        )

        # Initialize PufferEnv with buffers
        PufferEnv.__init__(self, buf=buf)

        # PufferLib buffer attributes (set by PufferEnv)
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray
        self.actions: np.ndarray

    # PufferLib required properties
    @property
    def single_observation_space(self):
        """Single agent observation space for PufferLib."""
        return self._observation_space

    @property
    def single_action_space(self):
        """Single agent action space for PufferLib."""
        return self._action_space

    @property
    def emulated(self) -> bool:
        """Native envs do not use emulation (PufferLib compatibility)."""
        return False

    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations and set up buffers.

        Returns:
            Initial observations array
        """
        # Get task config for reset
        assert self._task is not None, "Task not set"
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Reset the environment to get initial observations
        observations, _ = super().reset(game_config_dict)
        return observations

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        # Get new task from curriculum and its config
        assert self._curriculum is not None, "Curriculum not set"
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        return super().reset(game_config_dict, seed)

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            actions: Array of actions with shape (num_agents, 2) and dtype int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        # Call base step implementation
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Return PufferLib-compatible tuple
        return observations, rewards, terminals, truncations, infos

    # PufferLib required properties
    @property
    @override
    def done(self) -> bool:
        """Check if environment is done."""
        # Environment is always configured in this design
        return self._should_reset
