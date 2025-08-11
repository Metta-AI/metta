from __future__ import annotations

import random
from typing import Any

from metta.map.mapgen import MapGen, MapGenParams

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
        self._current_task = None

    def reset(self, *args, **kwargs):
        """Reset the environment with a new curriculum task.

        Gets a new task from the curriculum, configures the environment
        with the task's configuration, then calls the environment's reset method.
        """
        # Extract seed from kwargs if provided, otherwise use a random seed
        seed = kwargs.get("seed") or random.randint(0, 2**31 - 1)

        # Get new task from curriculum with seed
        self._current_task = self._curriculum.get_task(seed)

        # Get environment config from task
        env_config = self._current_task.get_env_config()

        # Generate LevelMap from MapGenParams if present in game.map_gen_params
        if hasattr(env_config, "game") and hasattr(env_config.game, "map_gen_params"):
            map_gen_params = env_config.game.map_gen_params

            # Create MapGen instance and build the level
            if isinstance(map_gen_params, dict):
                mapgen = MapGen(**map_gen_params)
            elif isinstance(map_gen_params, MapGenParams):
                mapgen = MapGen(**map_gen_params.model_dump())
            else:
                mapgen = MapGen(**map_gen_params)

            # Build the level and set it in the game config
            level_map = mapgen.build()
            env_config.game.level_map = level_map

            # Remove map_gen_params since we've converted it to level_map
            if hasattr(env_config.game, "map_gen_params"):
                delattr(env_config.game, "map_gen_params")

        # Configure environment with task configuration
        self._env.set_env_cfg(env_config)

        # Call environment reset
        return self._env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        """Step the environment and handle task completion.

        Calls the environment's step method, then checks if the episode is done
        and completes the current task with the curriculum if so.
        """
        # Call environment step
        result = self._env.step(*args, **kwargs)

        # Check if episode is done and complete task
        if len(result) >= 3:  # Assuming (obs, reward, done, ...) format
            done = result[2]
            if done and self._current_task is not None:
                # Extract reward as score for curriculum
                reward = result[1] if len(result) >= 2 else 0.0
                self._curriculum._record_task_completion(self._current_task, reward)

        return result

    def __getattr__(self, name: str):
        """Delegate all other attribute access to the wrapped environment."""
        return getattr(self._env, name)
