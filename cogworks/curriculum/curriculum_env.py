from __future__ import annotations

from typing import Any

from .curriculum import Curriculum


class CurriculumEnvWrapper:
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
            self._current_task.complete(mean_reward)
            self._current_task = self._curriculum.get_task()
            self._env.set_env_cfg(self._current_task.get_env_cfg())

        return obs, rewards, terminals, truncations, infos

    def __getattr__(self, name: str):
        """Delegate all other attribute access to the wrapped environment."""
        return getattr(self._env, name)
