from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import wandb
from omegaconf import DictConfig, OmegaConf

from .curriculum import Curriculum, Task
from .util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class StagedProgressiveCurriculum(Curriculum):
    """
    A curriculum that progresses through multiple stages of training.
    
    This replaces multi-stage training approaches by automatically transitioning
    between different curriculum stages based on performance or time thresholds.
    """

    def __init__(
        self,
        stages: List[Dict],
        env_overrides: Optional[DictConfig] = None,
        transition_criteria: str = "performance",  # "performance" or "time"
        performance_threshold: float = 0.8,
        time_threshold_steps: int = 100000,
        transition_smoothing: float = 0.1,
    ):
        """
        Initialize the staged progressive curriculum.
        
        Args:
            stages: List of stage configurations, each containing:
                - curriculum: curriculum config path or dict
                - name: stage name for logging
                - weight: initial weight for this stage (optional)
            env_overrides: Environment overrides to apply to all stages
            transition_criteria: How to determine stage transitions
            performance_threshold: Performance threshold for transitions
            time_threshold_steps: Time threshold for transitions (in steps)
            transition_smoothing: Smoothing factor for stage weights
        """
        self.stages = stages
        self.env_overrides = env_overrides or OmegaConf.create({})
        self.transition_criteria = transition_criteria
        self.performance_threshold = performance_threshold
        self.time_threshold_steps = time_threshold_steps
        self.transition_smoothing = transition_smoothing
        
        # Initialize stage curricula
        self._stage_curricula = []
        self._stage_names = []
        self._stage_weights = []
        
        for i, stage_config in enumerate(stages):
            if isinstance(stage_config, str):
                # Simple string config path
                curriculum = curriculum_from_config_path(stage_config, self.env_overrides)
                name = stage_config.split("/")[-1]
                weight = 1.0
            else:
                # Dict config with curriculum path and optional metadata
                curriculum = curriculum_from_config_path(stage_config["curriculum"], self.env_overrides)
                name = stage_config.get("name", f"stage_{i}")
                weight = stage_config.get("weight", 1.0)
            
            self._stage_curricula.append(curriculum)
            self._stage_names.append(name)
            self._stage_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self._stage_weights)
        if total_weight > 0:
            self._stage_weights = [w / total_weight for w in self._stage_weights]
        
        # Training state
        self._current_stage = 0
        self._stage_performance = [0.0] * len(stages)
        self._stage_completion_steps = [0] * len(stages)
        self._total_steps = 0
        self._stage_start_step = 0
        
        logger.info(f"StagedProgressiveCurriculum initialized with {len(stages)} stages")
        for i, (name, weight) in enumerate(zip(self._stage_names, self._stage_weights)):
            logger.info(f"  Stage {i}: {name} (weight: {weight:.3f})")

    def get_task(self) -> Task:
        """Get a task from the current stage curriculum."""
        current_curriculum = self._stage_curricula[self._current_stage]
        task = current_curriculum.get_task()
        
        # Add stage information to task
        task.add_parent(self, f"stage_{self._current_stage}")
        
        # Log current stage and stage probabilities
        if wandb.run is not None:
            stage_probs = {name: (1.0 if i == self._current_stage else 0.0) 
                          for i, name in enumerate(self._stage_names)}
            wandb.run.log({
                "curriculum/current_stage": self._current_stage,
                "curriculum/stage_name": self._stage_names[self._current_stage],
                "curriculum/stage_probs": stage_probs,
                "curriculum/stage_performance": dict(zip(self._stage_names, self._stage_performance)),
            }, commit=False)
        
        return task

    def complete_task(self, id: str, score: float):
        """Complete a task and potentially transition to next stage."""
        # Update performance for current stage
        self._stage_performance[self._current_stage] = (
            (1 - self.transition_smoothing) * self._stage_performance[self._current_stage] + 
            self.transition_smoothing * score
        )
        
        # Check for stage transition
        should_transition = self._should_transition_to_next_stage()
        
        if should_transition and self._current_stage < len(self.stages) - 1:
            self._transition_to_next_stage()
        
        # Complete task in current stage curriculum
        current_curriculum = self._stage_curricula[self._current_stage]
        current_curriculum.complete_task(id, score)

    def _should_transition_to_next_stage(self) -> bool:
        """Determine if we should transition to the next stage."""
        if self._current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_performance = self._stage_performance[self._current_stage]
        current_stage_steps = self._total_steps - self._stage_start_step
        
        if self.transition_criteria == "performance":
            return current_performance >= self.performance_threshold
        elif self.transition_criteria == "time":
            return current_stage_steps >= self.time_threshold_steps
        else:
            return False

    def _transition_to_next_stage(self):
        """Transition to the next stage."""
        old_stage = self._current_stage
        self._current_stage += 1
        
        # Log transition
        logger.info(
            f"Transitioning from stage {old_stage} ({self._stage_names[old_stage]}) "
            f"to stage {self._current_stage} ({self._stage_names[self._current_stage]}) "
            f"at step {self._total_steps}"
        )
        
        if wandb.run is not None:
            wandb.run.log({
                "curriculum/stage_transition": {
                    "from_stage": old_stage,
                    "to_stage": self._current_stage,
                    "from_name": self._stage_names[old_stage],
                    "to_name": self._stage_names[self._current_stage],
                    "step": self._total_steps,
                    "performance": self._stage_performance[old_stage],
                }
            }, commit=True)
        
        # Update stage tracking
        self._stage_completion_steps[old_stage] = self._total_steps - self._stage_start_step
        self._stage_start_step = self._total_steps

    def update_step_count(self, steps: int):
        """Update the total step count for time-based transitions."""
        self._total_steps = steps 