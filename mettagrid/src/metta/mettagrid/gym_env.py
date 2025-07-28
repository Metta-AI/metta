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

from metta.mettagrid.base_env import MettaGridEnv
from metta.mettagrid.curriculum.core import Curriculum
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
        # Call the base reset method from MettaGridEnv
        obs, info = super().reset(seed)

        # Handle single-agent return format
        if self._single_agent and obs is not None:
            return obs[0], info
        return obs, info

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

    # Gymnasium properties are inherited from base MettaGridEnv


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

    # Properties inherited from base MettaGridEnv
