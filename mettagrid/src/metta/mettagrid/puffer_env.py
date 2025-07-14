"""
MettaGridPufferEnv - PufferLib adapter for MettaGrid.

This class implements the PufferLib environment interface using the base MettaGridEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pufferlib import PufferEnv
from typing_extensions import override

from metta.mettagrid.base_env import MettaGridEnv
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

# Data types must match PufferLib -- see pufferlib/vector.py
dtype_observations = np.dtype(np.uint8)
dtype_terminals = np.dtype(bool)
dtype_truncations = np.dtype(bool)
dtype_rewards = np.dtype(np.float32)
dtype_actions = np.dtype(np.int32)


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
        # Initialize base environment
        MettaGridEnv.__init__(
            self,
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
            **kwargs,
        )

        # Create initial core environment so PufferEnv can access observation space
        self._core_env = self._create_core_env(0)

        # Initialize PufferEnv
        PufferEnv.__init__(self, buf)

        # PufferLib buffer attributes (set by PufferEnv)
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray
        self.actions: np.ndarray

    @override
    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations and set up buffers.

        Returns:
            Initial observations array
        """
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        # Validate buffer dtypes
        assert self.observations.dtype == dtype_observations
        assert self.terminals.dtype == dtype_terminals
        assert self.truncations.dtype == dtype_truncations
        assert self.rewards.dtype == dtype_rewards

        # Set buffers in core environment
        self._core_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        # Get initial observations
        return self._core_env.get_initial_observations()

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        return self.reset_base(seed)

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
        infos = self.step_base(actions)

        # Return PufferLib-compatible tuple
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    # PufferLib required properties
    @property
    @override
    def done(self) -> bool:
        """Check if environment is done."""
        return self._should_reset
