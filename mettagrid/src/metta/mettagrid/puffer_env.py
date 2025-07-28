"""
MettaGridPufferEnv - PufferLib adapter for MettaGrid.

This class implements the PufferLib environment interface using the base MettaGridEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pufferlib import PufferEnv
from typing_extensions import override

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

# Data types must match PufferLib -- see pufferlib/vector.py
dtype_observations = np.dtype(np.uint8)
dtype_terminals = np.dtype(bool)
dtype_truncations = np.dtype(bool)
dtype_rewards = np.dtype(np.float32)
dtype_actions = np.dtype(np.int32)
dtype_masks = np.dtype(bool)
dtype_success = np.dtype(bool)


class MettaGridPufferEnv(MettaGridEnv, PufferEnv):
    """
    PufferLib adapter for MettaGrid environments.

    This class combines the base MettaGridEnv functionality with PufferLib's
    vectorized environment interface, including proper buffer management.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        buf: Optional[Any] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize PufferLib environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            buf: PufferLib buffer object
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            **kwargs: Additional arguments
        """
        # Initialize base environment (this also calls PufferEnv.__init__ with buf)
        MettaGridEnv.__init__(
            self,
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            buf=buf,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
            **kwargs,
        )

        # Core environment is already created by base MettaGridEnv initialization
        # PufferEnv needs access to it for observation space
        self._core_env = self._c_env

        # Note: PufferEnv.__init__ is already called by MettaGridEnv.__init__
        # No need to call it again - this would cause double initialization

        # PufferLib buffer attributes (set by PufferEnv)
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray
        self.actions: np.ndarray

    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations and set up buffers.

        Returns:
            Initial observations array
        """
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        # Reset the environment to get initial observations
        observations, _ = super().reset()
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
        return super().reset(seed)

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
        return self._should_reset
