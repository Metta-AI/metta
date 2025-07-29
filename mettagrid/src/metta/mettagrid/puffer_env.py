"""
MettaGridPufferEnv - PufferLib adapter for MettaGrid.

This class implements the PufferLib environment interface using the base MettaGridEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pufferlib import PufferEnv
from typing_extensions import override

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level


class MettaGridPufferEnv(MettaGridCore, PufferEnv):
    """
    PufferLib adapter for MettaGrid environments.

    This class provides a clean PufferLib interface for users who want to use
    MettaGrid environments with their own PufferLib training setup.
    No training features are included - this is purely for PufferLib compatibility.
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
        Initialize PufferLib environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            buf: PufferLib buffer object
            **kwargs: Additional arguments
        """
        # Initialize core environment (no training features)
        MettaGridCore.__init__(
            self,
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            **kwargs,
        )

        # Initialize PufferEnv with buffers
        PufferEnv.__init__(self, buf=buf)

        # Core environment is already created by base MettaGridCore initialization
        # PufferEnv needs access to it for observation space
        self._core_env = self._c_env

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
