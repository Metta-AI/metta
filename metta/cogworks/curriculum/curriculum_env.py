from __future__ import annotations

from typing import Any

from pufferlib import PufferEnv

from .curriculum import Curriculum


class CurriculumEnv(PufferEnv):
    """Environment wrapper that integrates with a curriculum system.

    This wrapper passes all function calls to the wrapped environment, with special
    handling for reset() and step() methods to integrate with curriculum task management.
    """

    def __init__(self, env: Any, curriculum: Curriculum):
        """Initialize the curriculum environment wrapper.

        Args:
            env: The environment to wrap
            curriculum: The curriculum system to use for task generation
        """

        assert hasattr(env, "set_mg_config"), "Environment must have set_mg_config method"
        assert hasattr(env, "get_episode_rewards"), "Environment must have get_episode_rewards method"

        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
        self._env = env
        self._curriculum = curriculum
        self._current_task = self._curriculum.get_task()

        # Stats batching configuration - updating stats too frequently is an SPS hit
        self._stats_update_counter = 0
        self._stats_update_frequency = 50  # Batch stats updates to reduce overhead

        # Pre-compute string prefix for performance
        self._CURRICULUM_STAT_PREFIX = "env_curriculum/"

        # Track first reset to avoid hasattr checks
        self._first_reset_done = False

        # Cache for curriculum stats to avoid recomputation
        self._cached_stats = {}
        self._stats_cache_valid = False

    def _add_curriculum_stats_to_info(self, info_dict: dict) -> None:
        """Add curriculum statistics to info dictionary for logging.

        This method consolidates the curriculum stats logging logic to avoid duplication
        and enables batching of expensive stats calculations.
        """
        # Only update curriculum stats periodically to reduce overhead
        if self._stats_update_counter >= self._stats_update_frequency:
            if not self._stats_cache_valid:
                self._cached_stats = self._curriculum.stats()
                self._stats_cache_valid = True

            # Use pre-computed prefix for better performance
            for key, value in self._cached_stats.items():
                info_dict[self._CURRICULUM_STAT_PREFIX + key] = value
            self._stats_update_counter = 0

    def reset(self, *args, **kwargs):
        """Reset the environment and get a new task from curriculum."""
        obs, info = self._env.reset(*args, **kwargs)

        # Get a new task from curriculum
        self._current_task = self._curriculum.get_task()
        self._env.set_mg_config(self._current_task.get_env_cfg())

        # Invalidate stats cache on reset
        self._stats_cache_valid = False

        # Only log curriculum stats on reset if cache is invalid or this is first reset
        if not self._first_reset_done:
            curriculum_stats = self._curriculum.stats()
            for key, value in curriculum_stats.items():
                info[self._CURRICULUM_STAT_PREFIX + key] = value
            self._first_reset_done = True

        return obs, info

    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if the episode is done
        and completes the current task with the curriculum if so. Then gives the
        environment a new env config.
        """
        obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

        if terminals.all() or truncations.all():
            mean_reward = self._env.get_episode_rewards().mean()
            self._current_task.complete(mean_reward)
            # Update the curriculum algorithm with task performance for learning progress
            self._curriculum.update_task_performance(self._current_task._task_id, mean_reward)
            self._current_task = self._curriculum.get_task()
            self._env.set_mg_config(self._current_task.get_env_cfg())

            # Invalidate stats cache when task changes
            self._stats_cache_valid = False

        # Add curriculum stats to info for logging (batched)
        self._stats_update_counter += 1
        self._add_curriculum_stats_to_info(infos)

        return obs, rewards, terminals, truncations, infos

    def set_stats_update_frequency(self, frequency: int) -> None:
        """Set the frequency of curriculum stats updates during steps.

        Args:
            frequency: Number of steps between stats updates (default: 50)
        """
        self._stats_update_frequency = max(1, frequency)
        self._stats_update_counter = 0  # Reset counter

    def force_stats_update(self) -> None:
        """Force an immediate update of curriculum stats."""
        self._stats_update_counter = self._stats_update_frequency
        self._stats_cache_valid = False

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped environment when attribute not found.

        This is called only when the attribute is not found on CurriculumEnv itself,
        providing automatic delegation for all PufferEnv interface methods.
        """
        return getattr(self._env, name)
