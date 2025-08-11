"""
MettaGridPufferBase - Base PufferLib integration for MettaGrid.

This class provides PufferLib compatibility for MettaGrid environments by inheriting
from both MettaGridCore and PufferEnv. This allows MettaGrid environments to be used
directly with PufferLib training infrastructure.

Architecture:
- MettaGridPufferBase inherits from: MettaGridCore + PufferEnv
- MettaGridEnv inherits from: MettaGridPufferBase
- This enables MettaGridEnv to work seamlessly with PufferLib training code

For users:
- Use MettaGridEnv directly with PufferLib (it inherits PufferLib functionality)
- Alternatively, use PufferLib's MettaPuff wrapper for additional PufferLib features:
  https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments/metta/environment.py

This avoids double-wrapping while maintaining full PufferLib compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pufferlib import PufferEnv
from typing_extensions import override

from metta.mettagrid.config import EnvConfig
from metta.mettagrid.core import MettaGridCore
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
        env_config: EnvConfig,
        render_mode: Optional[str] = None,
        buf: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize PufferLib base environment.

        Args:
            env_config: Environment configuration
            render_mode: Rendering mode
            buf: PufferLib buffer object
            **kwargs: Additional arguments
        """

        # Initialize core environment
        MettaGridCore.__init__(
            self,
            env_config=env_config,
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
        return super().reset(env_config=None, seed=seed)

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
