"""
MettaGridPufferBase - Base PufferLib integration for MettaGrid.

This class provides PufferLib compatibility for MettaGrid environments by inheriting
from both MettaGridCore and PufferEnv. This allows MettaGrid environments to be used
directly with PufferLib training infrastructure.

Provides:
 - Auto-reset on episode completion
 - Persistent buffers for re-use between resets

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
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from pufferlib import PufferEnv

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
        mg_config: MettaGridConfig,
        render_mode: Optional[str] = None,
        buf: Optional[Any] = None,
    ):
        """
        Initialize PufferLib base environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built game map
            buf: PufferLib buffer object
            **kwargs: Additional arguments
        """
        # Initialize core environment. Do this first to set up observation space for PufferEnv.
        MettaGridCore.__init__(
            self,
            mg_config=mg_config,
            render_mode=render_mode,
        )

        # Initialize PufferEnv with buffers
        PufferEnv.__init__(self, buf=buf)

        # Auto-Reset
        self._should_reset = False

        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray
        self._last_sanitized_actions: Optional[np.ndarray] = None

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
        observations, _ = super().reset()
        return observations

    def _sanitize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Return a copy of ``actions`` clamped to valid type/argument ranges."""

        if actions.size == 0:
            return actions

        # Ensure we are working on an int32 copy to avoid mutating caller buffers
        sanitized = np.array(actions, dtype=dtype_actions, copy=True)

        if sanitized.shape[-1] < 2:
            return sanitized

        num_action_types = len(self.action_names)
        if num_action_types == 0:
            return sanitized

        # Normalize action types into valid range
        action_types = sanitized[..., 0]
        normalized_types = np.mod(action_types, num_action_types)
        sanitized[..., 0] = normalized_types

        # Clip action arguments to the per-action max argument value
        max_args = np.asarray(self.max_action_args, dtype=dtype_actions)
        max_args = np.clip(max_args, 0, None)
        allowed_args = max_args.take(normalized_types)
        sanitized[..., 1] = np.clip(sanitized[..., 1], 0, allowed_args)

        self._last_sanitized_actions = sanitized
        return sanitized

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._should_reset = False
        self._last_sanitized_actions = None

        if seed is not None:
            self._current_seed = seed

        observations, info = super().reset(seed)
        return observations, info

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        sanitized_actions = self._sanitize_actions(actions)
        observations, rewards, terminals, truncations, infos = super().step(sanitized_actions)
        if terminals.all() or truncations.all():
            self._should_reset = True

        return observations, rewards, terminals, truncations, infos

    # PufferLib required properties
    @property
    @override
    def done(self) -> bool:
        """Check if environment is done."""
        return self._should_reset
