"""
MettaGridGymEnv - Gymnasium adapter for MettaGrid.

This class implements the Gymnasium environment interface using the base MettaGridEnv.
Supports both single-agent and multi-agent modes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env as GymEnv
from gymnasium import spaces
from typing_extensions import override

from metta.mettagrid.base_env import MettaGridEnv
from metta.mettagrid.curriculum import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

# Data types for Gymnasium
dtype_observations = np.dtype(np.uint8)
dtype_terminals = np.dtype(bool)
dtype_truncations = np.dtype(bool)
dtype_rewards = np.dtype(np.float32)
dtype_actions = np.dtype(np.int32)


class MettaGridGymEnv(MettaGridEnv, GymEnv):
    """
    Gymnasium adapter for MettaGrid environments.

    This class provides a Gymnasium-compatible interface for MettaGrid environments,
    supporting both single-agent and multi-agent scenarios.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        single_agent: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Gymnasium environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            single_agent: Whether to use single-agent mode
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

        # Initialize Gym environment
        GymEnv.__init__(self)

        self._single_agent = single_agent

        # Create initial core environment for property access
        self._core_env = self._create_core_env(0)

        # Buffers for environment data
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None
        self._truncations: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None

    def _allocate_buffers(self) -> None:
        """Allocate buffers based on environment dimensions."""
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        num_agents = self._core_env.num_agents
        obs_space = self._core_env.observation_space

        # Allocate buffers
        self._observations = np.zeros((num_agents,) + obs_space.shape, dtype=dtype_observations)
        self._terminals = np.zeros(num_agents, dtype=dtype_terminals)
        self._truncations = np.zeros(num_agents, dtype=dtype_truncations)
        self._rewards = np.zeros(num_agents, dtype=dtype_rewards)

    @override
    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations and set up buffers.

        Returns:
            Initial observations array
        """
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        # Allocate buffers
        self._allocate_buffers()

        # Set buffers in core environment
        assert self._observations is not None
        assert self._terminals is not None
        assert self._truncations is not None
        assert self._rewards is not None
        self._core_env.set_buffers(self._observations, self._terminals, self._truncations, self._rewards)

        # Get initial observations
        obs = self._core_env.get_initial_observations()

        # Return single agent observation if in single agent mode
        if self._single_agent:
            return obs[0]

        return obs

    @override
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
        return self.reset_base(seed)

    @override
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
        infos = self.step_base(actions)

        # Get step results
        if self._observations is None or self._rewards is None or self._terminals is None or self._truncations is None:
            raise RuntimeError("Buffers not initialized")

        observations = self._observations.copy()
        rewards = self._rewards.copy()
        terminals = self._terminals.copy()
        truncations = self._truncations.copy()

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

    # Gymnasium required properties
    @property
    @override
    def observation_space(self) -> spaces.Space:
        """Get observation space."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")

        single_space = self._core_env.observation_space

        if self._single_agent:
            return single_space
        else:
            # Multi-agent space - return array of spaces
            return spaces.Tuple([single_space for _ in range(self._core_env.num_agents)])

    @property
    @override
    def action_space(self) -> spaces.Space:
        """Get action space."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")

        single_space = self._core_env.action_space

        if self._single_agent:
            return single_space
        else:
            # Multi-agent space - return array of spaces
            return spaces.Tuple([single_space for _ in range(self._core_env.num_agents)])


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
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize single-agent Gymnasium environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            **kwargs: Additional arguments
        """
        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
            single_agent=True,
            **kwargs,
        )

    @property
    @override
    def observation_space(self) -> spaces.Box:
        """Get single-agent observation space."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.observation_space

    @property
    @override
    def action_space(self) -> spaces.MultiDiscrete:
        """Get single-agent action space."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.action_space
