"""Periodic reset environment wrapper for in-context learning trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
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


class PeriodicResetEnv(PufferEnv):
    """Environment wrapper that periodically resets the environment while preserving LSTM state.

    This wrapper divides an episode into multiple trials by resetting the environment
    at fixed intervals, but hides the termination signals from the trainer to maintain
    LSTM state continuity. Adds a countdown timer to observations indicating time until
    next reset.
    """

    def __init__(self, env: Any, config: PeriodicResetConfig):
        """Initialize the periodic reset environment wrapper.

        Args:
            env: The environment to wrap
            config: Configuration for periodic resets
        """
        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
        self._env = env
        self._config = config
        self._trial_length = config.trial_length
        self._step_counter = 0
        self._time_to_reset = self._trial_length
        self._original_observation_space = None

    def _add_timer_to_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add time_to_reset counter to the observation."""
        if isinstance(obs, dict):
            # Copy the observation dict and add our timer
            augmented_obs = obs.copy()

            # Handle both single and multi-agent cases
            if "time_to_reset" not in augmented_obs:
                # For multi-agent, we need to broadcast the timer to all agents
                if hasattr(self._env, "num_agents"):
                    num_agents = self._env.num_agents
                    timer_array = np.full((num_agents,), self._time_to_reset, dtype=np.int32)
                else:
                    # Single agent case
                    timer_array = np.array([self._time_to_reset], dtype=np.int32)

                augmented_obs["time_to_reset"] = timer_array

            return augmented_obs
        else:
            # If obs is not a dict, we can't easily augment it
            # This shouldn't happen with MettaGrid environments
            return obs

    def reset(self, **kwargs):
        """Reset the environment and initialize counters."""
        obs, info = self._env.reset(**kwargs)
        self._step_counter = 0
        self._time_to_reset = self._trial_length

        # Add timer to initial observation
        obs = self._add_timer_to_observation(obs)

        return obs, info

    def step(self, actions):
        """Step the environment with periodic resets while preserving LSTM state."""
        # Step the underlying environment
        obs, rewards, terminals, truncations, infos = self._env.step(actions)

        # Update our counters
        self._step_counter += 1
        self._time_to_reset -= 1

        # Check if we need to do a periodic reset
        if self._time_to_reset <= 0 and self._config.number_of_trials_in_episode > 1:
            # Reset the environment internally (new locations, new task state)
            # but don't reset the episode counters
            reset_obs, _ = self._env.reset()

            # Use the reset observation but keep the original rewards/terminals
            obs = reset_obs

            # Reset the trial timer but keep the episode step counter
            self._time_to_reset = self._trial_length

            # Critically: Don't change terminals/truncations to preserve LSTM state
            # The trainer will see this as a continuous episode

        # Add timer to observation
        obs = self._add_timer_to_observation(obs)

        return obs, rewards, terminals, truncations, infos

    def __getattribute__(self, name: str):
        """Intercept all attribute access and delegate to wrapped environment when appropriate."""
        # First, handle our own attributes to avoid infinite recursion
        if name in (
            "_env",
            "_config",
            "_trial_length",
            "_step_counter",
            "_time_to_reset",
            "_original_observation_space",
            "_add_timer_to_observation",
            "step",
            "reset",
        ):
            return object.__getattribute__(self, name)

        # Try to get the attribute from our wrapped environment
        try:
            env = object.__getattribute__(self, "_env")
            return getattr(env, name)
        except AttributeError:
            # If not found in wrapped env, fall back to parent class
            return object.__getattribute__(self, name)
