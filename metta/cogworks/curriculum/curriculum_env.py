from __future__ import annotations

from typing import Any

from .curriculum import Curriculum


class CurriculumEnv:
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
        self._env = env
        self._curriculum = curriculum
        self._current_task = self._curriculum.get_task()

    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if the episode is done
        and completes the current task with the curriculum if so. Then gives the
        environment a new env config.
        """
        obs, rewards, terminals, truncations, infos = self._env.step(*args, **kwargs)

        if len(terminals) > 0 and (terminals.all() or truncations.all()):
            # Handle empty rewards case
            mean_reward = rewards.mean() if len(rewards) > 0 else 0.0

            # Calculate task-scaled performance if reward_target is available
            task_scaled_performance = None
            env_cfg = self._current_task.get_env_cfg()
            if hasattr(env_cfg, "reward_target") and env_cfg.reward_target is not None:
                reward_target = env_cfg.reward_target
                if reward_target > 0:
                    task_scaled_performance = min(mean_reward / reward_target, 1.0)

            # Complete task with raw reward (scaled performance will be handled separately)
            self._current_task.complete(mean_reward)

            # Store scaled performance for logging if available
            if task_scaled_performance is not None:
                # Add to infos for logging
                if "task_scaled_performance" not in infos:
                    infos["task_scaled_performance"] = {}
                infos["task_scaled_performance"][self._current_task._task_id] = task_scaled_performance

            # Get new task
            self._current_task = self._curriculum.get_task()
            self._env.set_env_cfg(self._current_task.get_env_cfg())

        return obs, rewards, terminals, truncations, infos

    def __getattr__(self, name: str):
        """Delegate all other attribute access to the wrapped environment."""
        return getattr(self._env, name)
