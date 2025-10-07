"""
MettaGridGymEnv - Gymnasium adapter for MettaGrid.

This class implements the Gymnasium environment interface using the base MettaGridEnv.
Supports both single-agent and multi-agent modes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env as GymEnv
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions


class MettaGridGymEnv(MettaGridCore, GymEnv):
    """
    Gymnasium adapter for MettaGrid environments.

    This class provides a Gymnasium-compatible interface for MettaGrid environments,
    supporting both single-agent and multi-agent scenarios.
    No training features are included - this is purely for Gymnasium compatibility.

    Inherits from:
    - MettaGridCore: Core C++ environment wrapper functionality
    - gymnasium.Env: Standard Gymnasium environment interface
      https://github.com/Farama-Foundation/Gymnasium/blob/ad23dfbbe29f83107404f9f6a56131f6b498d0d7/gymnasium/core.py#L23
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Gymnasium environment.

        Args:
            mg_config: Environment configuration
            render_mode: Rendering mode
        """
        assert mg_config.game.num_agents == 1, "Gymnasium environments must be single-agent"

        # Initialize core functionality
        MettaGridCore.__init__(
            self,
            mg_config,
            render_mode=render_mode,
        )

        # Initialize Gym environment
        GymEnv.__init__(self)

    @override  # gymnasium.Env.reset
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observations, info)
        """
        # Call the base reset method
        obs, info = super().reset(seed)

        return obs[0], info

    @override  # gymnasium.Env.step
    def step(self, action: np.ndarray | int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            action: Discrete action index for the single agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_array = np.asarray(action, dtype=dtype_actions)
        if action_array.ndim == 0:
            actions = action_array.reshape(1)
        elif action_array.ndim == 1 and action_array.shape[0] == 1:
            actions = action_array
        else:
            raise ValueError(f"Expected scalar action for single-agent gym env, received shape {action_array.shape}")

        # Call base step implementation
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Handle single-agent return format
        return observations[0], rewards[0].item(), terminals[0].item(), truncations[0].item(), infos

    # Gymnasium space properties
    @property
    @override  # gymnasium.Env.observation_space
    def observation_space(self):
        """Get observation space."""
        return self._observation_space

    @property
    @override  # gymnasium.Env.action_space
    def action_space(self):
        """Get action space."""
        return self._action_space

    # PufferLib compatibility properties
    @property
    def single_observation_space(self):
        """Single agent observation space (PufferLib compatibility)."""
        return self._observation_space

    @property
    def single_action_space(self):
        """Single agent action space (PufferLib compatibility)."""
        return self._action_space
