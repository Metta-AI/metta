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

        # Per-label metrics tracking
        self._per_label_lp_scores = {}
        self._per_label_completion_counts = {}  # Cumulative across all training
        self._per_label_completion_counts_last_epoch = {}  # For computing deltas

        # Shared aggregated counts across all environments (set by reset_epoch_counters)
        self._aggregated_epoch_counts = None

    def reset_epoch_counters(self, aggregated_counts: dict[str, int] | None = None) -> None:
        """Reset per-epoch tracking at the start of a new epoch.

        This ensures that per-epoch delta calculations start fresh for each epoch,
        preventing completions from one epoch leaking into the next.

        Args:
            aggregated_counts: Optional aggregated counts across all environments.
                If provided, this will be used for logging instead of per-env counts.
        """
        # Reset the baseline for delta calculations
        self._per_label_completion_counts_last_epoch = self._per_label_completion_counts.copy()

        # Store aggregated counts if provided
        self._aggregated_epoch_counts = aggregated_counts

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

            # Add per-label learning progress metrics
            if self._per_label_lp_scores:
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_lp_scores"] = self._per_label_lp_scores.copy()

            # Report per-epoch deltas (derivative) instead of cumulative counts
            # Use aggregated counts if available (summed across all envs), otherwise use local counts
            if self._aggregated_epoch_counts is not None:
                # Log aggregated counts (already summed across all environments)
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_samples_this_epoch"] = (
                    self._aggregated_epoch_counts.copy()
                )
            elif self._per_label_completion_counts:
                # Fallback to local counts if aggregation hasn't happened yet
                per_epoch_counts = {}
                for label, cumulative_count in self._per_label_completion_counts.items():
                    last_count = self._per_label_completion_counts_last_epoch.get(label, 0)
                    per_epoch_counts[label] = cumulative_count - last_count

                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_samples_this_epoch"] = per_epoch_counts

            # Also log cumulative for reference (optional, can be removed if not needed)
            if self._per_label_completion_counts:
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_cumulative_samples"] = (
                    self._per_label_completion_counts.copy()
                )

            # NOTE: Do NOT reset the baseline here - only reset at epoch boundaries
            # via reset_epoch_counters() to get accurate per-epoch counts

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

            # Update per-label metrics tracking
            label = self._current_task.get_label()
            # Only track if label is a valid string (not None, not a Mock)
            if label is not None and isinstance(label, str):
                # Update LP score with EMA (α = 0.01)
                lp_score = self._curriculum.get_task_lp_score(self._current_task._task_id)
                if label in self._per_label_lp_scores:
                    self._per_label_lp_scores[label] = 0.99 * self._per_label_lp_scores[label] + 0.01 * lp_score
                else:
                    self._per_label_lp_scores[label] = lp_score

                # Update cumulative completion counts
                self._per_label_completion_counts[label] = self._per_label_completion_counts.get(label, 0) + 1

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

    def __getattribute__(self, name: str):
        """Intercept all attribute access and delegate to wrapped environment when appropriate.
        This handles the case where PufferEnv defines methods that raise NotImplementedError,
        ensuring they get properly delegated to the wrapped environment.
        """
        # First, handle our own attributes to avoid infinite recursion
        if name in (
            "_env",
            "_curriculum",
            "_current_task",
            "step",
            "_add_curriculum_stats_to_info",
            "_stats_update_counter",
            "_stats_update_frequency",
            "set_stats_update_frequency",
            "force_stats_update",
            "_cached_stats",
            "_stats_cache_valid",
            "_first_reset_done",
            "_per_label_lp_scores",
            "_per_label_completion_counts",
            "_per_label_completion_counts_last_epoch",
            "_aggregated_epoch_counts",
            "reset_epoch_counters",
            "_CURRICULUM_STAT_PREFIX",
        ):
            return object.__getattribute__(self, name)

        # Try to get the attribute from our wrapped environment
        try:
            env = object.__getattribute__(self, "_env")
            return getattr(env, name)
        except AttributeError:
            # If not found in wrapped env, fall back to parent class
            return object.__getattribute__(self, name)
