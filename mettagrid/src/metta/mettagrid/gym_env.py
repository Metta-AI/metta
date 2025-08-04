"""
MettaGridGymEnv - Gymnasium adapter for MettaGrid.

This class implements the Gymnasium environment interface using the base MettaGridEnv.
Supports both single-agent and multi-agent modes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env as GymEnv
from omegaconf import OmegaConf
from typing_extensions import override

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level

# Data types for Gymnasium - import from C++ module
from metta.mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)


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
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        single_agent: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Gymnasium environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            single_agent: Whether to use single-agent mode
            **kwargs: Additional arguments
        """
        # Get level from curriculum if not provided
        if level is None:
            task = curriculum.get_task()
            level = task.env_cfg().game.map_builder.build()

        # Ensure we have a level
        assert level is not None, "Level must be provided or generated from curriculum"

        # Store curriculum for reset operations
        self._curriculum = curriculum
        self._task = self._curriculum.get_task()

        # Get game config for core initialization
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Initialize core functionality
        MettaGridCore.__init__(
            self,
            level=level,
            game_config_dict=game_config_dict,
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
        # Get new task from curriculum and its config
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Call the base reset method
        obs, info = super().reset(game_config_dict, seed)

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
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        **kwargs: Any,
    ):
        """
        Initialize single-agent Gymnasium environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            **kwargs: Additional arguments
        """
        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            single_agent=True,
            **kwargs,
        )
