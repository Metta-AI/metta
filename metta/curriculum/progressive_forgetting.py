"""Progressive curriculum for measuring catastrophic forgetting with smooth task switching."""

import logging
import random
from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.progressive import ProgressiveMultiTaskCurriculum

logger = logging.getLogger(__name__)


class ProgressiveForgettingCurriculum(ProgressiveMultiTaskCurriculum):
    """Curriculum that implements smooth switching between task sets to measure catastrophic forgetting.

    This curriculum uses a progress-based approach where:
    - All tasks in task set A are positioned at progress point 0.0
    - All tasks in task set B are positioned at progress point 1.0
    - A small gating function slope creates smooth transitions between task sets
    - Training continues on one task set until performance threshold is reached
    """

        def __init__(
        self,
        task_sets: Dict[str, Dict[str, float]],
        env_overrides: Optional[DictConfig] = None,
        performance_threshold: float = 0.8,
        smoothing: float = 0.1,
        blending_smoothness: float = 0.2,  # Moderate value for smooth transitions
        randomize_order: bool = True,
    ):
        """Initialize the progressive forgetting curriculum.

        Args:
            task_sets: Dictionary mapping task set names to task dictionaries
            env_overrides: Environment overrides
            performance_threshold: Performance threshold to trigger task switching
            smoothing: Smoothing factor for performance tracking
            blending_smoothness: Smoothness of task transitions (small = sharp, large = smooth)
            randomize_order: Whether to randomize the order of task sets
        """
        # Flatten all tasks and assign weights
        all_tasks = {}
        self.task_set_mapping = {}

        task_set_names = list(task_sets.keys())
        if randomize_order:
            random.shuffle(task_set_names)

        # Assign equal weights to all tasks within each set
        for task_set_name in task_set_names:
            tasks = task_sets[task_set_name]
            weight_per_task = 1.0 / len(tasks)

            for task_name in tasks:
                all_tasks[task_name] = weight_per_task
                self.task_set_mapping[task_name] = task_set_name

        # Store parameters for later initialization
        self._env_overrides = env_overrides or DictConfig({})
        self._performance_threshold = performance_threshold
        self._smoothing = smoothing
        self._progression_rate = 0.005  # Slower progression for smoother transitions
        self._progression_mode = "perf"
        self._blending_smoothness = blending_smoothness
        self._blending_mode = "logistic"

        # Initialize curricula without validation (bypass MultiTaskCurriculum validation)
        self._curricula = {t: self._curriculum_from_id(t) for t in all_tasks.keys()}
        self._task_weights = all_tasks
        self._task_order = list(all_tasks.keys())

        # Initialize progressive curriculum state
        self._progress = 0.0
        self._smoothed_performance = 0.0
        self._step_count = 0
        self._last_score = None
        self._update_progressive_weights()

        self.task_sets = task_sets
        self.task_set_order = task_set_names
        self.task_set_performance = {name: 0.0 for name in task_sets.keys()}
        self._task_scores = {}  # Track individual task scores

        logger.info(f"Initialized ProgressiveForgettingCurriculum with {len(task_sets)} task sets")
        logger.info(f"Task set order: {self.task_set_order}")
        logger.info(f"Blending smoothness: {blending_smoothness}")

    def _curriculum_from_id(self, cfg_path: str):
        """Create curriculum from config path without agent validation."""
        from metta.mettagrid.curriculum.util import curriculum_from_config_path
        return curriculum_from_config_path(cfg_path, self._env_overrides)

    def _evaluate_task_set_performance(self):
        """Evaluate performance on all task sets."""
        for task_set_name in self.task_sets:
            # Calculate average performance for this task set
            task_set_tasks = self.task_sets[task_set_name]
            performances = []

            for task_name in task_set_tasks:
                if task_name in self._task_scores:
                    performances.append(self._task_scores[task_name])

            if performances:
                avg_performance = np.mean(performances)
                # Update with smoothing
                old_perf = self.task_set_performance[task_set_name]
                self.task_set_performance[task_set_name] = (
                    self._smoothing * avg_performance + (1 - self._smoothing) * old_perf
                )

        logger.info(f"Task set performances: {self.task_set_performance}")

    def complete_task(self, id: str, score: float):
        """Complete a task and update curriculum state."""
        # Track individual task scores
        self._task_scores[id] = score

        # Evaluate task set performance periodically
        if len(self._task_scores) % 10 == 0:  # Every 10 tasks
            self._evaluate_task_set_performance()

        # Call parent method to update progress and weights
        super().complete_task(id, score)

    def get_curriculum_stats(self) -> Dict[str, float]:
        """Return curriculum statistics for logging."""
        # Convert task set name to numeric index for stats compatibility
        current_task_set_idx = 0
        if self._progress < 0.5:
            # In first task set
            current_task_set_idx = 0
        else:
            # In second task set
            current_task_set_idx = 1

        stats = {
            "current_task_set": float(current_task_set_idx),
            "progress": self._progress,
            "smoothed_performance": self._smoothed_performance,
        }

        # Add performance stats for each task set
        for task_set_name, performance in self.task_set_performance.items():
            stats[f"perf_{task_set_name}"] = performance

        return stats
