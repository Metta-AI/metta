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

from metta.mettagrid.core import MettaGridCore

# Data types for Gymnasium - import from C++ module
from metta.mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.mettagrid_config import EnvConfig


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
        env_config: EnvConfig,
        render_mode: Optional[str] = None,
        single_agent: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Gymnasium environment.

        Args:
            env_config: Environment configuration
            render_mode: Rendering mode
            single_agent: Whether to use single-agent mode
            **kwargs: Additional arguments
        """
        # Store env_config for reset operations
        self._env_config = env_config

        # Initialize core functionality with EnvConfig
        MettaGridCore.__init__(
            self,
            env_cfg=env_config,
            render_mode=render_mode,
            **kwargs,
        )

        # Initialize Gym environment
        GymEnv.__init__(self)

        self._single_agent = single_agent

        # Initialize buffer attributes for memory management
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None
        self._truncations: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None

        # Allocate buffers for C++ environment
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate numpy arrays for C++ environment shared memory."""
        # Allocate observation buffer
        obs_shape = (self.num_agents, *self._observation_space.shape)
        self._observations = np.zeros(obs_shape, dtype=dtype_observations)

        # Allocate terminal/truncation/reward buffers
        self._terminals = np.zeros(self.num_agents, dtype=dtype_terminals)
        self._truncations = np.zeros(self.num_agents, dtype=dtype_truncations)
        self._rewards = np.zeros(self.num_agents, dtype=dtype_rewards)

        # Set buffers in C++ environment for direct writes
        self._c_env_instance.set_buffers(self._observations, self._terminals, self._truncations, self._rewards)

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
        # Get env config from options or use default
        if options and "env_config" in options:
            env_cfg = options["env_config"]
            # Ensure it's an EnvConfig object
            if not isinstance(env_cfg, EnvConfig):
                raise TypeError(f"env_config must be an EnvConfig object, got {type(env_cfg)}")
        else:
            env_cfg = self._env_config

        # Call the base reset method with EnvConfig
        obs, info = super().reset(env_cfg, seed)

        # Handle single-agent return format
        if self._single_agent and obs is not None:
            return obs[0], info
        return obs, info

    @override  # gymnasium.Env.step
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            action: Action array. For single-agent: shape (2,). For multi-agent: shape (num_agents, 2)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Handle single-agent action format
        if self._single_agent:
            if action.ndim == 1:
                # Convert single action to multi-agent format
                actions = action[np.newaxis, ...]  # Add batch dimension
            else:
                actions = action
        else:
            actions = action

        # Ensure correct dtype
        actions = actions.astype(dtype_actions)

        # Call base step implementation
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Handle single-agent return format
        if self._single_agent:
            return (
                observations[0],  # Single agent observation
                float(rewards[0]),  # Single agent reward
                bool(terminals[0]),  # Single agent terminal
                bool(truncations[0]),  # Single agent truncation
                infos,
            )
        else:
            # Multi-agent format - return arrays
            return (observations, rewards, terminals, truncations, infos)

    # Gymnasium space properties
    @property
    @override  # gymnasium.Env.observation_space
    def observation_space(self):
        """Get observation space."""
        if self._single_agent:
            return self._observation_space
        else:
            # Multi-agent case - return the multi-agent space
            return self._observation_space

    @property
    @override  # gymnasium.Env.action_space
    def action_space(self):
        """Get action space."""
        if self._single_agent:
            return self._action_space
        else:
            # Multi-agent case - return the multi-agent space
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


class SingleAgentMettaGridGymEnv(MettaGridGymEnv):
    """
    Single-agent wrapper for MettaGrid Gymnasium environments.

    This is a convenience class that automatically sets single_agent=True
    and provides a cleaner interface for single-agent use cases.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize single-agent Gymnasium environment.

        Args:
            env_config: Environment configuration
            render_mode: Rendering mode
            **kwargs: Additional arguments
        """
        super().__init__(
            env_config=env_config,
            render_mode=render_mode,
            single_agent=True,
            **kwargs,
        )
