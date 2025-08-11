"""
CurriculumEnv - Wrapper for AutoResetEnv that adds curriculum support.

This class wraps AutoResetEnv and handles curriculum-based task selection,
converting curriculum tasks to EnvConfig for the underlying environment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from metta.mettagrid.auto_reset_env import AutoResetEnv
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import LevelMap
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter


class CurriculumEnv(AutoResetEnv):
    """
    Wrapper for AutoResetEnv that adds curriculum support.

    This class handles:
    - Getting tasks from curriculum
    - Converting task configs to EnvConfig
    - Passing EnvConfig to AutoResetEnv
    - Managing task completion
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[LevelMap] = None,
        buf: Optional[Any] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize CurriculumEnv wrapper.

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
        self._curriculum = curriculum
        self._current_task = None
        self._level = level

        # Get initial task and convert to EnvConfig
        self._current_task = self._curriculum.get_task()
        env_config = self._current_task.env_config()

        # Initialize base class with the environment config
        super().__init__(
            env_config=env_config,
            render_mode=render_mode,
            buf=buf,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
            **kwargs,
        )

        # Store references for curriculum management
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._is_training = is_training

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with a new task from curriculum.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        # Get next task from curriculum
        self._current_task = self._curriculum.get_task()
        env_config = self._current_task.env_config()

        # Update environment config (set_env_cfg will assert episode is finished)
        # This ensures we don't accidentally change config mid-episode
        self.set_env_cfg(env_config)

        # Call parent's reset which will use the updated env_config
        return super().reset(seed=seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            action: Action array

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Handle task completion (terminated and truncated are arrays)
        if (
            (isinstance(terminated, np.ndarray) and terminated.any())
            or (isinstance(truncated, np.ndarray) and truncated.any())
            or (not isinstance(terminated, np.ndarray) and terminated)
            or (not isinstance(truncated, np.ndarray) and truncated)
        ):
            if self._current_task:
                # Calculate score from cumulative reward or info
                score = info.get("episode_return", reward if not isinstance(reward, np.ndarray) else reward.sum())
                self._current_task.complete_trial(score)

        return obs, reward, terminated, truncated, info
