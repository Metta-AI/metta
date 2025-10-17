"""PufferEnv integration for curriculum learning.

This module requires pufferlib-core to be installed:
    pip install pufferlib-core

Or as an optional dependency:
    pip install agora[pufferlib]
"""

from __future__ import annotations

from typing import Any, Generic

try:
    from pufferlib import PufferEnv
except ImportError as e:
    raise ImportError(
        "pufferlib-core is required for CurriculumEnv. "
        "Install with: pip install pufferlib-core or pip install agora[pufferlib]"
    ) from e

from agora.config import TConfig
from agora.curriculum import Curriculum


class CurriculumEnv(PufferEnv, Generic[TConfig]):
    """Environment wrapper that integrates with a curriculum system.

    This wrapper passes all function calls to the wrapped environment, with special
    handling for reset() and step() methods to integrate with curriculum task management.

    The wrapped environment must implement two methods:
    - set_task_config(config: TConfig): Apply a new task configuration
    - get_episode_rewards() -> array-like: Return rewards for completed episodes

    Example:
        >>> from agora import Curriculum, CurriculumConfig, SingleTaskGenerator
        >>> from agora.wrappers import CurriculumEnv
        >>>
        >>> # Create curriculum
        >>> task_gen_config = SingleTaskGenerator.Config(env=my_config)
        >>> curriculum_config = CurriculumConfig(task_generator=task_gen_config)
        >>> curriculum = curriculum_config.make()
        >>>
        >>> # Wrap environment
        >>> env = MyPufferEnv(...)
        >>> curriculum_env = CurriculumEnv(env, curriculum)
    """

    def __init__(self, env: Any, curriculum: Curriculum[TConfig], task_config_setter: str = "set_mg_config"):
        """Initialize the curriculum environment wrapper.

        Args:
            env: The environment to wrap (must have set_mg_config/set_task_config and get_episode_rewards methods)
            curriculum: The curriculum system to use for task generation
            task_config_setter: Name of method to call to set task config (default: "set_mg_config" for backward compat)

        Raises:
            AssertionError: If environment doesn't have required methods
        """
        # Try set_mg_config first for backward compatibility, fall back to set_task_config
        if not hasattr(env, task_config_setter):
            if hasattr(env, "set_mg_config"):
                task_config_setter = "set_mg_config"
            elif hasattr(env, "set_task_config"):
                task_config_setter = "set_task_config"
            else:
                raise AssertionError(
                    f"Environment must have {task_config_setter}, set_mg_config, or set_task_config method"
                )
        assert hasattr(env, "get_episode_rewards"), "Environment must have get_episode_rewards method"

        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
        self._env = env
        self._curriculum = curriculum
        self._task_config_setter = task_config_setter
        self._current_task = self._curriculum.get_task()

        # Stats batching configuration - updating stats too frequently is an SPS hit
        self._stats_update_counter = 0
        self._stats_update_frequency = 50  # Batch stats updates to reduce overhead

        # Pre-compute string prefix for performance
        self._CURRICULUM_STAT_PREFIX = "env_curriculum/"

        # Track first reset to avoid hasattr checks
        self._first_reset_done = False

        # Cache for curriculum stats to avoid recomputation
        self._cached_stats: dict[str, float] = {}
        self._stats_cache_valid = False

        # Per-label metrics tracking
        self._per_label_lp_scores: dict[str, float] = {}
        self._per_label_completion_counts: dict[str, int] = {}  # Cumulative across all training
        self._per_label_completion_counts_last_epoch: dict[str, int] = {}  # For computing deltas

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch tracking at the start of a new epoch.

        This ensures that per-epoch delta calculations start fresh for each epoch,
        preventing completions from one epoch leaking into the next.
        """
        # Reset the baseline for delta calculations
        self._per_label_completion_counts_last_epoch = self._per_label_completion_counts.copy()

    def _add_curriculum_stats_to_info(self, info_dict: dict[str, Any]) -> None:
        """Add curriculum statistics to info dictionary for logging.

        This method consolidates the curriculum stats logging logic to avoid duplication
        and enables batching of expensive stats calculations.

        Args:
            info_dict: Dictionary to add stats to (modified in-place)
        """
        # Only update curriculum stats periodically to reduce overhead
        if self._stats_update_counter >= self._stats_update_frequency:
            if not self._stats_cache_valid:
                self._cached_stats = self._curriculum.stats()
                self._stats_cache_valid = True

            # Use pre-computed prefix for better performance
            for key, value in self._cached_stats.items():
                info_dict[self._CURRICULUM_STAT_PREFIX + key] = value

            # Add per-label learning progress metrics
            if self._per_label_lp_scores:
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_lp_scores"] = self._per_label_lp_scores.copy()

            # Report per-epoch deltas (derivative) instead of cumulative counts
            if self._per_label_completion_counts:
                # Compute delta since last epoch
                per_epoch_counts = {}
                for label, cumulative_count in self._per_label_completion_counts.items():
                    last_count = self._per_label_completion_counts_last_epoch.get(label, 0)
                    per_epoch_counts[label] = cumulative_count - last_count

                # Log the per-epoch deltas (this is what you want to visualize)
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_samples_this_epoch"] = per_epoch_counts

                # Also log cumulative for reference (optional, can be removed if not needed)
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_cumulative_samples"] = (
                    self._per_label_completion_counts.copy()
                )

                # Update last epoch counts for next delta computation
                self._per_label_completion_counts_last_epoch = self._per_label_completion_counts.copy()

            self._stats_update_counter = 0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and get a new task from curriculum.

        Returns:
            Tuple of (observation, info_dict)
        """
        obs, info = self._env.reset(*args, **kwargs)

        # Get a new task from curriculum
        self._current_task = self._curriculum.get_task()
        getattr(self._env, self._task_config_setter)(self._current_task.get_env_cfg())

        # Invalidate stats cache on reset
        self._stats_cache_valid = False

        # Only log curriculum stats on reset if cache is invalid or this is first reset
        if not self._first_reset_done:
            curriculum_stats = self._curriculum.stats()
            for key, value in curriculum_stats.items():
                info[self._CURRICULUM_STAT_PREFIX + key] = value
            self._first_reset_done = True

        return obs, info

    def step(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if the episode is done
        and completes the current task with the curriculum if so. Then gives the
        environment a new task config.

        Returns:
            Tuple of (obs, rewards, terminals, truncations, infos)
        """
        obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

        if terminals.all() or truncations.all():
            mean_reward = self._env.get_episode_rewards().mean()
            self._current_task.complete(mean_reward)

            # Update the curriculum algorithm with task performance for learning progress
            self._curriculum.update_task_performance(self._current_task._task_id, mean_reward)

            # Track per-label learning progress scores
            task_label = self._current_task.get_label()
            if hasattr(self._curriculum, "_algorithm") and self._curriculum._algorithm is not None:
                # Get LP score from algorithm if available
                if hasattr(self._curriculum._algorithm, "get_learning_progress_score"):
                    lp_score = self._curriculum._algorithm.get_learning_progress_score(  # type: ignore[attr-defined]
                        self._current_task._task_id
                    )
                    # Update per-label LP score (exponential moving average)
                    if task_label in self._per_label_lp_scores:
                        self._per_label_lp_scores[task_label] = (
                            0.9 * self._per_label_lp_scores[task_label] + 0.1 * lp_score
                        )
                    else:
                        self._per_label_lp_scores[task_label] = lp_score

                    # Track completion counts per label
                    self._per_label_completion_counts[task_label] = (
                        self._per_label_completion_counts.get(task_label, 0) + 1
                    )

            self._current_task = self._curriculum.get_task()
            getattr(self._env, self._task_config_setter)(self._current_task.get_env_cfg())

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

    def __getattribute__(self, name: str) -> Any:
        """Intercept all attribute access and delegate to wrapped environment when appropriate.

        This handles the case where PufferEnv defines methods that raise NotImplementedError,
        ensuring they get properly delegated to the wrapped environment.

        Args:
            name: Attribute name to access

        Returns:
            Attribute value
        """
        # First, handle our own attributes to avoid infinite recursion
        if name in (
            "_env",
            "_curriculum",
            "_current_task",
            "_task_config_setter",
            "step",
            "reset",
            "_add_curriculum_stats_to_info",
            "_stats_update_counter",
            "_stats_update_frequency",
            "set_stats_update_frequency",
            "force_stats_update",
            "reset_epoch_counters",
            "_cached_stats",
            "_stats_cache_valid",
            "_first_reset_done",
            "_CURRICULUM_STAT_PREFIX",
            "_per_label_lp_scores",
            "_per_label_completion_counts",
            "_per_label_completion_counts_last_epoch",
        ):
            return object.__getattribute__(self, name)

        # Try to get the attribute from our wrapped environment
        try:
            env = object.__getattribute__(self, "_env")
            return getattr(env, name)
        except AttributeError:
            # If not found in wrapped env, fall back to parent class
            return object.__getattribute__(self, name)


__all__ = ["CurriculumEnv"]
