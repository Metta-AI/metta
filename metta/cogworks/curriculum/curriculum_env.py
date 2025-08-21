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

        assert hasattr(env, "set_env_config"), "Environment must have set_env_config method"
        assert hasattr(env, "get_episode_rewards"), "Environment must have get_episode_rewards method"

        # We don't call super().__init__() because this wrapper
        # proxies all calls to the wrapped environment.
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

        if terminals.all() or truncations.all():
            mean_reward = self._env.get_episode_rewards().mean()
            self._current_task.complete(mean_reward)
            self._current_task = self._curriculum.get_task()
            self._env.set_env_config(self._current_task.get_env_cfg())

        return obs, rewards, terminals, truncations, infos

    # Override PufferEnv abstract methods to avoid NotImplementedError
    def reset(self, *args, **kwargs):
        """Reset the environment."""
        return self._env.reset(*args, **kwargs)
    
    def close(self):
        """Close the environment."""
        return self._env.close()
    
    def render(self, *args, **kwargs):
        """Render the environment."""
        return self._env.render(*args, **kwargs)
    
    def async_reset(self, *args, **kwargs):
        """Async reset for the environment."""
        return self._env.async_reset(*args, **kwargs)
    
    def send(self, *args, **kwargs):
        """Send actions to the environment."""
        return self._env.send(*args, **kwargs)
    
    def recv(self):
        """Receive observations from the environment."""
        return self._env.recv()
    
    # Delegate properties
    @property
    def single_observation_space(self):
        return self._env.single_observation_space
    
    @property
    def single_action_space(self):
        return self._env.single_action_space
    
    @property
    def num_agents(self):
        return self._env.num_agents
    
    @property
    def agent_per_batch(self):
        return self._env.agent_per_batch
    
    @property
    def emulated(self):
        return self._env.emulated
    
    @property
    def done(self):
        return self._env.done
    
    @property
    def driver_env(self):
        return self._env.driver_env
    
    def __getattr__(self, name: str):
        """Delegate any other attribute access to wrapped environment.

        This is only called for attributes not found on this object,
        providing a fallback for any additional env-specific attributes.
        """
        return getattr(self._env, name)
