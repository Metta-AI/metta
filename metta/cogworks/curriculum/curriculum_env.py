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

import logging

from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from pufferlib import PufferEnv

from .curriculum import Curriculum

logger = logging.getLogger(__name__)


class CurriculumEnv(PufferEnv):
    """Environment wrapper that integrates with a curriculum system.

    This wrapper:
    - Handles task selection and completion through the curriculum
    - ALWAYS emits per-label episode counts (needed for basic curriculum monitoring)
    - All other curriculum stats are collected centrally at epoch boundaries by StatsReporter
    """

    def __init__(self, env: MettaGridPufferEnv, curriculum: Curriculum):
        """Initialize the curriculum environment wrapper.

        Args:
            env: The environment to wrap (currently typed as MettaGridPufferEnv)
            curriculum: The curriculum system to use for task generation

        TODO: For more flexibility and type safety, replace MettaGridPufferEnv with a Protocol
        that defines the required interface (set_mg_config, get_episode_rewards, reset, step).
        This would enable structural typing and support any environment implementing these methods.
        """

        assert hasattr(env, "set_mg_config"), "Environment must have set_mg_config method"
        assert hasattr(env, "get_episode_rewards"), "Environment must have get_episode_rewards method"

        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
        self._env: MettaGridPufferEnv = env
        self._curriculum = curriculum
        self._current_task = self._curriculum.get_task()

        # Check if troubleshooting logging is enabled
        self._enable_per_label_tracking = getattr(
            curriculum._algorithm.hypers, "show_curriculum_troubleshooting_logging", False
        )

    def _get_task_with_retries(self, context: str = "") -> None:
        """Get a new task from curriculum with retry logic for invalid configurations.

        Args:
            context: Optional context string for logging (e.g., "on reset", "on episode completion")

        Raises:
            AssertionError/ValueError: If all retries are exhausted
        """
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
                    context_msg = f" {context}" if context else ""
                    logger.warning(
                        f"Task configuration error{context_msg} "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. Resampling new task..."
                    )
                    # Mark the task as invalid so it can be evicted
                    if hasattr(self._current_task, "_task_id"):
                        self._current_task.complete(-1.0)  # Mark as failed
                    continue
                else:
                    # All retries exhausted - re-raise the error
                    context_msg = f" {context}" if context else ""
                    logger.error(
                        f"Failed to find valid task configuration after {max_retries} attempts{context_msg}. "
                        f"Last error: {e}"
                    )
                    raise

    def reset(self, *args, **kwargs):
        """Reset the environment and get a new task from curriculum."""
        # Get a new task from curriculum, with retry logic for invalid configurations
        self._get_task_with_retries(context="on reset")

        obs, info = self._env.reset(*args, **kwargs)
        return obs, info

    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if ANY episode is done
        and completes the current task with the curriculum. Note: Only counts one
        completion per environment episode, not per agent (even if multiple agents
        complete simultaneously).
        """
        obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

        # Handle completion for ANY environment that finished (not all)
        # NOTE: Count ONCE per environment episode, not once per agent
        if terminals.any() or truncations.any():
            # Get per-environment rewards for completed episodes
            episode_rewards = self._env.get_episode_rewards()

            # Get evictions and add to info dict (for gini calculation)
            # NOTE: Use get_evictions() NOT get_and_reset_evictions()
            # The reset should only happen at epoch boundaries (in training loop),
            # not on every episode completion.
            evictions = self._curriculum.get_evictions()
            if evictions:
                if "env_curriculum_stats/per_label_evictions" not in infos:
                    infos["env_curriculum_stats/per_label_evictions"] = {}
                for label, count in evictions.items():
                    infos["env_curriculum_stats/per_label_evictions"][label] = (
                        infos["env_curriculum_stats/per_label_evictions"].get(label, 0) + count
                    )

            # Calculate mean reward across all agents in this environment
            # All agents in the same environment complete simultaneously, so take mean
            mean_reward = float(episode_rewards.mean())

            # Debug logging for reward tracking (can be removed after verification)
            logger.debug(
                f"Task {self._current_task._task_id} (label={self._current_task.get_label()}) "
                f"completed with mean_reward={mean_reward:.4f} "
                f"(raw episode_rewards: min={episode_rewards.min():.4f}, "
                f"max={episode_rewards.max():.4f}, mean={episode_rewards.mean():.4f})"
            )

            # Record ONE completion for this environment episode
            self._current_task.complete(mean_reward)
            self._curriculum.update_task_performance(self._current_task._task_id, mean_reward)

            # ALWAYS emit per-label sample count (needed for basic curriculum monitoring)
            # Note: get_label() always returns a string (enforced in CurriculumTask.__init__)
            label = self._current_task.get_label()
            if "env_curriculum_stats/per_label_samples" not in infos:
                infos["env_curriculum_stats/per_label_samples"] = {}
            infos["env_curriculum_stats/per_label_samples"][label] = (
                infos["env_curriculum_stats/per_label_samples"].get(label, 0) + 1
            )

            # Get new task with retry logic for invalid configurations
            self._get_task_with_retries(context="on episode completion")

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
            "_get_task_with_retries",
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
