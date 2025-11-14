"""Environment wrapper for curriculum integration.

This module bridges the training loop and curriculum system. CurriculumEnv wraps any
environment and automatically handles task selection, application, and performance reporting.

Key responsibilities:
- Request new tasks from curriculum on each episode reset
- Apply task configuration to the wrapped environment
- Report episode outcomes back to curriculum for learning progress tracking
- Retry logic for invalid task configurations
- Optional per-label episode tracking for debugging

Why separate file: Keeps environment integration concerns separate from curriculum logic.
The curriculum is agnostic to how tasks are actually used - this wrapper implements the
"glue" that connects curriculum decisions to actual environment execution.
"""

from __future__ import annotations

from typing import Any

from pufferlib import PufferEnv

from .curriculum import Curriculum


class CurriculumEnv(PufferEnv):
    """Environment wrapper that integrates with a curriculum system.

    This wrapper:
    - Handles task selection and completion through the curriculum
    - ALWAYS emits per-label episode counts (needed for basic curriculum monitoring)
    - Optionally tracks first 3 tasks for debugging (if show_curriculum_troubleshooting_logging=True)
    - All other curriculum stats are collected centrally at epoch boundaries by StatsReporter
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

        # Check if troubleshooting logging is enabled
        self._enable_per_label_tracking = False
        if curriculum._algorithm is not None:
            self._enable_per_label_tracking = curriculum._algorithm.hypers.show_curriculum_troubleshooting_logging

        # Tracked task attributes (only if troubleshooting enabled)
        if self._enable_per_label_tracking:
            self._tracked_task_ids = []
            self._tracked_task_completions_this_epoch = {}
            self._tracked_task_completions_baseline = {}
        else:
            self._tracked_task_ids = None
            self._tracked_task_completions_this_epoch = None
            self._tracked_task_completions_baseline = None

    def reset(self, *args, **kwargs):
        """Reset the environment and get a new task from curriculum."""
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

        obs, info = self._env.reset(*args, **kwargs)
        return obs, info

    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if ANY episode is done
        and completes the current task with the curriculum for each completion.
        """
        obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

        # DEBUG: Track completions with detailed logging
        if not hasattr(self, "_debug_completion_count"):
            self._debug_completion_count = 0

        # Handle completion for ANY environment that finished (not all)
        if terminals.any() or truncations.any():
            self._debug_completion_count += 1

            # Log details on first 5 completions to see what's happening
            if self._debug_completion_count <= 5:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("=" * 80)
                logger.info(f"CURRICULUM_ENV: EPISODE COMPLETION #{self._debug_completion_count}")
                logger.info("=" * 80)
                logger.info(f"  terminals.shape: {terminals.shape}, terminals.any(): {terminals.any()}")
                logger.info(f"  truncations.shape: {truncations.shape}, truncations.any(): {truncations.any()}")
                logger.info(f"  Number of completions this step: {terminals.sum() + truncations.sum()}")

                # Show which envs completed
                completed_indices = [i for i, (t, tr) in enumerate(zip(terminals, truncations, strict=True)) if t or tr]
                logger.info(f"  Completed env indices: {completed_indices[:10]}...")
            # Get per-environment rewards for completed episodes
            episode_rewards = self._env.get_episode_rewards()

            # Get per-epoch evictions and add to info dict (for gini calculation)
            # This is called once per step when any environment completes
            evictions_this_epoch = self._curriculum.get_and_reset_evictions_this_epoch()
            if evictions_this_epoch:
                if "env_curriculum_stats/per_label_evictions_this_epoch" not in infos:
                    infos["env_curriculum_stats/per_label_evictions_this_epoch"] = {}
                for label, count in evictions_this_epoch.items():
                    infos["env_curriculum_stats/per_label_evictions_this_epoch"][label] = (
                        infos["env_curriculum_stats/per_label_evictions_this_epoch"].get(label, 0) + count
                    )

            # Update curriculum for each completed environment
            for env_idx, (term, trunc) in enumerate(zip(terminals, truncations, strict=True)):
                if term or trunc:
                    reward = float(episode_rewards[env_idx])

                    # DEBUG: Log first few reward updates
                    if self._debug_completion_count <= 5:
                        import logging

                        logger = logging.getLogger(__name__)
                        task_id = self._current_task._task_id
                        logger.info(f"  Env {env_idx}: reward={reward:.4f}, task_id={task_id}")
                        logger.info(f"    Calling curriculum.update_task_performance({task_id}, {reward:.4f})")

                    self._current_task.complete(reward)
                    # Update the curriculum algorithm with task performance for learning progress
                    self._curriculum.update_task_performance(self._current_task._task_id, reward)

                    # DEBUG: Verify update happened
                    if self._debug_completion_count <= 5:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info("    Update complete")

                    # ALWAYS emit per-label sample count (needed for basic curriculum monitoring)
                    label = self._current_task.get_label()
                    if label is not None and isinstance(label, str):
                        if "env_curriculum_stats/per_label_samples_this_epoch" not in infos:
                            infos["env_curriculum_stats/per_label_samples_this_epoch"] = {}
                        infos["env_curriculum_stats/per_label_samples_this_epoch"][label] = (
                            infos["env_curriculum_stats/per_label_samples_this_epoch"].get(label, 0) + 1
                        )

                    # Track task completions for troubleshooting (ONLY if flag enabled)
                    if self._enable_per_label_tracking:
                        task_id = self._current_task._task_id
                        if task_id not in self._tracked_task_ids and len(self._tracked_task_ids) < 3:
                            self._tracked_task_ids.append(task_id)

                        if task_id in self._tracked_task_ids:
                            self._tracked_task_completions_this_epoch[task_id] = (
                                self._tracked_task_completions_this_epoch.get(task_id, 0) + 1
                            )

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

            # DEBUG: Close completion log
            if self._debug_completion_count <= 5:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("=" * 80)

            # DEBUG: Stop after 5 completions to inspect logs
            if self._debug_completion_count == 5:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("")
                logger.info("=" * 80)
                logger.info("DEBUG: Stopping training after 5 episode completions for log inspection")
                logger.info("=" * 80)
                raise SystemExit("Debug stop: 5 episode completions reached")

        return obs, rewards, terminals, truncations, infos

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
            "_enable_per_label_tracking",
            "_tracked_task_ids",
            "_tracked_task_completions_this_epoch",
            "_tracked_task_completions_baseline",
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
