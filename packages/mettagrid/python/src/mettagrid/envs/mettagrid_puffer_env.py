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
# xcxc update docs

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete
from typing_extensions import override

from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.simulator import Simulator
from pufferlib.pufferlib import PufferEnv

# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)


class MettaGridPufferEnv(PufferEnv):
    """
    Wraps the Simulator class to provide PufferLib compatibility.

    Inherits from pufferlib.PufferEnv: High-performance vectorized environment interface
      https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments.py
    """

    def __init__(self, simulator: Simulator, buf: Any = None):  # xcxc
        self._simulator = simulator
        super().__init__(buf=buf)
        self.emulated: bool = False

        # Auto-Reset
        self._should_reset = False
        self.single_observation_space: Box = self._simulator._observation_space
        self.single_action_space: Discrete = self._simulator._action_space

    def _get_initial_observations(self) -> np.ndarray:
        observations, _ = super().reset()
        return observations

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._should_reset = False

        if seed is not None:
            self._current_seed = seed

        observations, info = super().reset(seed)
        return observations, info

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        observations, rewards, terminals, truncations, infos = super().step(actions)
        if terminals.all() or truncations.all():
            self._should_reset = True

        return observations, rewards, terminals, truncations, infos

    @property
    def observations(self) -> np.ndarray:
        return self.observations

    @observations.setter
    def observations(self, observations: np.ndarray) -> None:
        self._simulator._buffers.observations = observations

    @property
    def rewards(self) -> np.ndarray:
        return self.rewards

    @rewards.setter
    def rewards(self, rewards: np.ndarray) -> None:
        self._simulator._buffers.rewards = rewards

    @property
    def terminals(self) -> np.ndarray:
        return self.terminals

    @terminals.setter
    def terminals(self, terminals: np.ndarray) -> None:
        self._simulator._buffers.terminals = terminals

    @property
    def truncations(self) -> np.ndarray:
        return self.truncations

    @truncations.setter
    def truncations(self, truncations: np.ndarray) -> None:
        self._simulator._buffers.truncations = truncations

    @property
    def masks(self) -> np.ndarray:
        return self.masks

    @masks.setter
    def masks(self, masks: np.ndarray) -> None:
        self._simulator._buffers.masks = masks

    @property
    def actions(self) -> np.ndarray:
        return self.actions

    @actions.setter
    def actions(self, actions: np.ndarray) -> None:
        self._simulator._buffers.actions = actions
