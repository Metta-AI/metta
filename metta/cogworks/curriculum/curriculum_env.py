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
        self._CURRICULUM_STAT_PREFIX = "curriculum_stats/"

        # Track first reset to avoid hasattr checks
        self._first_reset_done = False

        # Per-label metrics tracking
        self._per_label_lp_scores = {}

        # Cache curriculum stats
        self._cached_curriculum_stats = {}
        self._curriculum_stats_cache_valid = False

        # Track first 3 tasks for detailed dynamics analysis
        self._tracked_task_ids = []  # Will store first 3 task IDs we encounter
        self._tracked_task_completions_this_epoch = {}  # task_id -> count this epoch
        self._tracked_task_completions_baseline = {}  # task_id -> count at last epoch reset

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch tracking at the start of a new epoch.

        Note: Per-label counts are emitted per-episode in infos (not accumulated),
        so this is a no-op except for tracked task completion counters.
        """
        # Reset tracked task completion baselines for per-epoch counting
        self._tracked_task_completions_baseline = self._tracked_task_completions_this_epoch.copy()

    def _add_curriculum_stats_to_info(self, info_dict: dict) -> None:
        """Add curriculum statistics to info dictionary for logging.

        Logs:
        - Per-label LP scores (EMA smoothed)
        - Pool composition fractions (fraction of task pool per label)
        - Total completions, evictions
        - Gini coefficient for sampling distribution across labels
        - Mean LP score in task pool
        """
        # Only update curriculum stats periodically to reduce overhead
        if self._stats_update_counter >= self._stats_update_frequency:
            # Get curriculum stats (with caching to reduce overhead)
            if not self._curriculum_stats_cache_valid:
                self._cached_curriculum_stats = self._curriculum.stats()
                self._curriculum_stats_cache_valid = True

            stats = self._cached_curriculum_stats

            # Add per-label learning progress metrics (EMA smoothed per environment)
            if self._per_label_lp_scores:
                info_dict[self._CURRICULUM_STAT_PREFIX + "per_label_lp_scores"] = self._per_label_lp_scores.copy()

            # Add pool composition fractions (fraction of task pool for each label)
            pool_composition = {}
            total_pool_size = stats.get("num_active_tasks", 0)
            if total_pool_size > 0:
                for key, value in stats.items():
                    if key.startswith("algorithm/pool_composition/"):
                        label = key.replace("algorithm/pool_composition/", "")
                        pool_composition[label] = value / total_pool_size

            if pool_composition:
                info_dict[self._CURRICULUM_STAT_PREFIX + "pool_composition_fraction"] = pool_composition

            # Add total completions
            if "num_completed" in stats:
                info_dict[self._CURRICULUM_STAT_PREFIX + "total_completions"] = stats["num_completed"]

            # Add number of evictions
            if "num_evicted" in stats:
                info_dict[self._CURRICULUM_STAT_PREFIX + "num_evicted"] = stats["num_evicted"]

            # Calculate and add Gini coefficient for sampling distribution
            sampling_counts = []
            for key, value in stats.items():
                if key.startswith("algorithm/sampling_counts/"):
                    sampling_counts.append(value)

            if sampling_counts:
                gini = self._calculate_gini_coefficient(sampling_counts)
                info_dict[self._CURRICULUM_STAT_PREFIX + "sampling_gini"] = gini

            # Calculate and add Gini coefficient for pool occupancy (completion counts)
            pool_completion_counts = []
            for key, value in stats.items():
                if key.startswith("algorithm/completion_counts/"):
                    pool_completion_counts.append(value)

            if pool_completion_counts:
                pool_gini = self._calculate_gini_coefficient(pool_completion_counts)
                info_dict[self._CURRICULUM_STAT_PREFIX + "pool_occupancy_gini"] = pool_gini

            # Calculate and add Gini coefficient for pool LP scores
            pool_lp_scores = []
            for key, value in stats.items():
                if key.startswith("algorithm/lp_scores/"):
                    pool_lp_scores.append(value)

            if pool_lp_scores:
                lp_gini = self._calculate_gini_coefficient(pool_lp_scores)
                info_dict[self._CURRICULUM_STAT_PREFIX + "pool_lp_gini"] = lp_gini

            # Add mean LP score from task pool
            if "algorithm/mean_lp_score" in stats:
                info_dict[self._CURRICULUM_STAT_PREFIX + "mean_pool_lp_score"] = stats["algorithm/mean_lp_score"]

            # Add tracked task dynamics (first 3 tasks encountered)
            if self._tracked_task_ids:
                tracked_lp_scores = {}
                tracked_completions_this_epoch = {}

                for i, task_id in enumerate(self._tracked_task_ids):
                    # Get LP score for this task
                    lp_score = self._curriculum.get_task_lp_score(task_id)
                    tracked_lp_scores[f"task_{i}"] = lp_score

                    # Get completion count this epoch (delta from baseline)
                    current_count = self._tracked_task_completions_this_epoch.get(task_id, 0)
                    baseline_count = self._tracked_task_completions_baseline.get(task_id, 0)
                    tracked_completions_this_epoch[f"task_{i}"] = current_count - baseline_count

                info_dict[self._CURRICULUM_STAT_PREFIX + "tracked_task_lp_scores"] = tracked_lp_scores
                info_dict[self._CURRICULUM_STAT_PREFIX + "tracked_task_completions_this_epoch"] = (
                    tracked_completions_this_epoch
                )

            self._stats_update_counter = 0

    def _calculate_gini_coefficient(self, values: list[float]) -> float:
        """Calculate Gini coefficient for a distribution.

        Measures inequality in sampling across labels:
        - 0 = perfect equality (all labels sampled equally)
        - 1 = perfect inequality (all samples from one label)

        Args:
            values: List of counts/frequencies (e.g., sampling counts per label)

        Returns:
            Gini coefficient between 0 and 1
        """
        if not values or len(values) == 0:
            return 0.0

        # Handle case with all zeros
        if sum(values) == 0:
            return 0.0

        # Sort values in ascending order
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini coefficient using the formula:
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        cumsum = 0.0
        for i, value in enumerate(sorted_values, start=1):
            cumsum += i * value

        total = sum(sorted_values)
        gini = (2.0 * cumsum) / (n * total) - (n + 1.0) / n

        return gini

    def reset(self, *args, **kwargs):
        """Reset the environment and get a new task from curriculum."""
        obs, info = self._env.reset(*args, **kwargs)

        # Invalidate curriculum stats cache on reset (task pool may have changed)
        self._curriculum_stats_cache_valid = False

        # Get a new task from curriculum, with retry logic for invalid configurations
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self._current_task = self._curriculum.get_task()
                self._env.set_mg_config(self._current_task.get_env_cfg())
                break  # Success - exit retry loop
            except (AssertionError, ValueError) as e:
                # Handle configuration errors (e.g., agent count mismatch, map too small)
                if attempt < max_retries - 1:
                    # Log warning and try a new task
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Task configuration error (attempt {attempt + 1}/{max_retries}): {e}. Resampling new task..."
                    )
                    # Mark the task as invalid so it can be evicted
                    if hasattr(self._current_task, "_task_id"):
                        self._current_task.complete(-1.0)  # Mark as failed
                    continue
                else:
                    # All retries exhausted - re-raise the error
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"Failed to find valid task configuration after {max_retries} attempts. Last error: {e}"
                    )
                    raise

        # Mark that first reset is done (for future use)
        if not self._first_reset_done:
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

            # Invalidate curriculum stats cache on task completion (stats have changed)
            self._curriculum_stats_cache_valid = False

            # Track first 3 task IDs for detailed dynamics analysis
            task_id = self._current_task._task_id
            if task_id not in self._tracked_task_ids and len(self._tracked_task_ids) < 3:
                self._tracked_task_ids.append(task_id)

            # Update completion counts for tracked tasks
            if task_id in self._tracked_task_ids:
                self._tracked_task_completions_this_epoch[task_id] = (
                    self._tracked_task_completions_this_epoch.get(task_id, 0) + 1
                )

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

                # Emit per-label completion count directly in infos (following episode stats pattern)
                # This will be summed across all vectorized environments automatically
                if "curriculum_stats/per_label_samples_this_epoch" not in infos:
                    infos["curriculum_stats/per_label_samples_this_epoch"] = {}
                infos["curriculum_stats/per_label_samples_this_epoch"][label] = 1

            # Get new task with retry logic for invalid configurations
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    self._current_task = self._curriculum.get_task()
                    self._env.set_mg_config(self._current_task.get_env_cfg())
                    break  # Success - exit retry loop
                except (AssertionError, ValueError) as e:
                    # Handle configuration errors (e.g., agent count mismatch, map too small)
                    if attempt < max_retries - 1:
                        # Log warning and try a new task
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Task configuration error on episode completion "
                            f"(attempt {attempt + 1}/{max_retries}): {e}. Resampling new task..."
                        )
                        # Mark the task as invalid so it can be evicted
                        if hasattr(self._current_task, "_task_id"):
                            self._current_task.complete(-1.0)  # Mark as failed
                        continue
                    else:
                        # All retries exhausted - re-raise the error
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"Failed to find valid task configuration after {max_retries} attempts "
                            f"on episode completion. Last error: {e}"
                        )
                        raise

        # Add label-specific curriculum stats to info for logging (batched)
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
            "_first_reset_done",
            "_per_label_lp_scores",
            "_cached_curriculum_stats",
            "_curriculum_stats_cache_valid",
            "_tracked_task_ids",
            "_tracked_task_completions_this_epoch",
            "_tracked_task_completions_baseline",
            "_calculate_gini_coefficient",
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
