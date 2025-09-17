"""Periodic reset environment wrapper for in-context learning trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, SupportsFloat, Tuple

import numpy as np
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from pufferlib import PufferEnv


@dataclass
class PeriodicResetConfig:
    """Configuration for periodic reset environment wrapper."""

    number_of_trials_in_episode: int = 1
    episode_length: int = 256

    @property
    def trial_length(self) -> int:
        """Calculate the length of each trial."""
        return self.episode_length // self.number_of_trials_in_episode

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.number_of_trials_in_episode < 1:
            raise ValueError("number_of_trials_in_episode must be >= 1")
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        if self.trial_length < 1:
            raise ValueError(
                f"trial_length ({self.trial_length}) must be >= 1. "
                f"Either increase episode_length ({self.episode_length}) or "
                f"decrease number_of_trials_in_episode ({self.number_of_trials_in_episode})"
            )


class PeriodicResetEnv(PufferEnv):
    """Environment wrapper that periodically resets the environment while preserving LSTM state.

    This wrapper divides an episode into multiple trials by resetting the environment
    at fixed intervals, but hides the termination signals from the trainer to maintain
    LSTM state continuity. Optionally adds a countdown timer to observations.

    Key Design Principles:
    - Preserves LSTM state by never exposing termination signals
    - Clean separation of trial logic from observation augmentation
    - Proper inheritance from PufferEnv
    - Clear configuration validation
    """

    def __init__(
        self,
        env: Env,
        config: PeriodicResetConfig,
        add_timer_to_obs: bool = True,
    ):
        """Initialize the periodic reset environment wrapper.

        Args:
            env: The environment to wrap
            config: Configuration for periodic resets
            add_timer_to_obs: Whether to add time_to_reset to observations
        """
        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
        self._env = env
        self._config = config
        self._add_timer_to_obs = add_timer_to_obs
        self._trial_step = 0
        self._total_step = 0

    @property
    def unwrapped(self) -> Env:
        """Return the base environment."""
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

    @property
    def time_to_reset(self) -> int:
        """Time remaining until next periodic reset."""
        return self._config.trial_length - self._trial_step

    def _should_periodic_reset(self) -> bool:
        """Check if we should perform a periodic reset."""
        return self._config.number_of_trials_in_episode > 1 and self._trial_step >= self._config.trial_length

    def _augment_observation(self, obs: ObsType) -> ObsType:
        """Add timer information to observation if enabled."""
        if not self._add_timer_to_obs or not isinstance(obs, dict):
            return obs

        # Copy observation and add timer
        augmented_obs = obs.copy()

        # Handle both single and multi-agent cases
        if hasattr(self._env, "num_agents") and self._env.num_agents > 1:
            timer_array = np.full((self._env.num_agents,), self.time_to_reset, dtype=np.int32)
        else:
            timer_array = np.array([self.time_to_reset], dtype=np.int32)

        augmented_obs["time_to_reset"] = timer_array
        return augmented_obs

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment and initialize counters."""
        obs, info = self._env.reset(**kwargs)
        self._trial_step = 0
        self._total_step = 0
        return self._augment_observation(obs), info

    def step(self, actions: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment with periodic resets while preserving LSTM state."""
        # Step the underlying environment
        obs, rewards, terminated, truncated, infos = self._env.step(actions)

        # Update counters
        self._trial_step += 1
        self._total_step += 1

        # Perform periodic reset if needed
        if self._should_periodic_reset():
            # Reset environment state but preserve episode continuity
            reset_obs, _ = self._env.reset()
            obs = reset_obs
            self._trial_step = 0

            # Critical: Never expose termination signals to preserve LSTM state

        return self._augment_observation(obs), rewards, terminated, truncated, infos

    def __getattribute__(self, name: str):
        """Intercept all attribute access and delegate to wrapped environment when appropriate.

        This handles the case where PufferEnv defines methods that raise NotImplementedError,
        ensuring they get properly delegated to the wrapped environment.
        """
        # First, handle our own attributes to avoid infinite recursion
        if name in (
            "_env",
            "_config",
            "_add_timer_to_obs",
            "_trial_step",
            "_total_step",
            "unwrapped",
            "time_to_reset",
            "_should_periodic_reset",
            "_augment_observation",
            "reset",
            "step",
        ):
            return object.__getattribute__(self, name)

        # Try to get the attribute from our wrapped environment
        try:
            env = object.__getattribute__(self, "_env")
            return getattr(env, name)
        except AttributeError:
            # If not found in wrapped env, fall back to parent class
            return object.__getattribute__(self, name)
