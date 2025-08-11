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
        self.env = env
        self.curriculum = curriculum
        self.current_task = None
        
    def reset(self, *args, **kwargs):
        """Reset the environment with a new curriculum task.
        
        Gets a new task from the curriculum, configures the environment
        with the task's configuration, then calls the environment's reset method.
        """
        # Extract seed from kwargs if provided, otherwise use a random seed
        import random
        seed = kwargs.get('seed') or random.randint(0, 2**31 - 1)
        
        # Get new task from curriculum with seed
        self.current_task = self.curriculum.get_task(seed)
        
        # Configure environment with task configuration
        self.env.set_env_cfg(self.current_task.get_env_config())
        
        # Call environment reset
        return self.env.reset(*args, **kwargs)
        
    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.
        
        Calls the environment's step method, then checks if the episode is done
        and completes the current task with the curriculum if so.
        """
        # Call environment step
        result = self.env.step(*args, **kwargs)
        
        # Check if episode is done and complete task
        if len(result) >= 3:  # Assuming (obs, reward, done, ...) format
            done = result[2]
            if done and self.current_task is not None:
                # Extract reward as score for curriculum
                reward = result[1] if len(result) >= 2 else 0.0
                self.curriculum.complete_task(self.current_task, reward)
        
        return result
        
    def __getattr__(self, name: str):
        """Delegate all other attribute access to the wrapped environment."""
        return getattr(self.env, name)