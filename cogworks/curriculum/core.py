"""Legacy curriculum module for backward compatibility.

This module provides the old curriculum interface for existing code and configurations
that haven't been migrated to the new Pydantic-based system.
"""

# Import the new curriculum classes but expose them with the old interface names
from .curriculum import Curriculum, Task

# For backward compatibility, we also need to provide the old classes
# The old SingleTrialTask and SingleTaskCurriculum were in the original core.py
# These are needed for some legacy tests and configurations

import hydra
from typing import List, Tuple
from omegaconf import DictConfig


class SingleTrialTask(Task):
    """Legacy SingleTrialTask for backward compatibility."""
    
    def __init__(self, id: str, curriculum, env_cfg: DictConfig):
        # Convert DictConfig to EnvConfig for the new system
        from metta.rl.env_config import EnvConfig
        if isinstance(env_cfg, DictConfig):
            # This is a legacy configuration, try to convert it
            env_config = EnvConfig()  # Use defaults, legacy configs may not convert cleanly
        else:
            env_config = env_cfg
            
        super().__init__(env_config, task_id=id)
        self._id = id
        self._name = id
        self._curricula = [(curriculum, id)]
        self._is_complete = False
        self._total_score = 0.0
        self._num_trials = 1  # Legacy default
        self._current_trial = 0
        self._env_cfg = env_config
        self._original_env_cfg = env_cfg

    def complete_trial(self, score: float):
        """Legacy method for completing trials."""
        assert not self._is_complete, "Task is already complete"
        self._current_trial += 1
        self._total_score += score
        if self._current_trial >= self._num_trials:
            self._is_complete = True
            for curriculum, id in self._curricula:
                curriculum.complete_task(id, self._total_score)

    def is_complete(self):
        """True if the task is complete, false otherwise."""
        return self._is_complete

    def env_cfg(self) -> DictConfig:
        """Returns the environment configuration for the current trial."""
        return self._original_env_cfg

    def original_env_cfg(self) -> DictConfig:
        """Returns the original environment configuration."""
        return self._original_env_cfg

    def id(self) -> str:
        """Returns the id of the task."""
        return self._id

    def name(self) -> str:
        """Returns the name of the task."""
        return self._name

    def short_name(self) -> str:
        """Returns the short name of the task."""
        return self.name().split("/")[-1]

    def add_parent(self, parent_curriculum, parent_id: str):
        """Adds a parent to the task."""
        self._curricula.append((parent_curriculum, parent_id))
        self._name = f"{parent_id}:{self._name}"


class SingleTaskCurriculum(Curriculum):
    """Legacy SingleTaskCurriculum for backward compatibility."""

    def __init__(self, task_id: str, task_cfg: DictConfig):
        # For legacy compatibility, we need to create a minimal curriculum config
        from .config import CurriculumConfig, WeightedTaskSetConfig
        from metta.rl.env_config import EnvConfig
        
        self._task_id = task_id
        self._task_cfg = task_cfg
        
        # Create a minimal task set config with the legacy env config
        env_config = EnvConfig()  # Use defaults for legacy configs
        task_set_config = WeightedTaskSetConfig(
            items=[],  # No items for single task
            overrides=None
        )
        curriculum_config = CurriculumConfig(task_set_config=task_set_config)
        
        super().__init__(curriculum_config, seed=42)

    def get_task(self, seed: int):
        """Get the single task."""
        return SingleTrialTask(self._task_id, self, self._task_cfg)

    def get_task_probs(self) -> dict[str, float]:
        """Return task probabilities."""
        return {self._task_id: 1.0}