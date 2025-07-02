"""Progressive curriculum for measuring catastrophic forgetting with sharp task switching."""

import logging
import random
from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)


class ProgressiveForgettingCurriculum(RandomCurriculum):
    """Curriculum that implements sharp switching between task sets to measure catastrophic forgetting.

    This curriculum trains on one task set until a performance threshold is reached,
    then sharply switches to another task set. It tracks performance on both task sets
    to measure forgetting and transfer learning.
    """

    def __init__(
        self,
        task_sets: Dict[str, Dict[str, float]],
        env_overrides: Optional[DictConfig] = None,
        performance_threshold: float = 0.8,
        smoothing: float = 0.1,
        switch_interval: int = 1000,
        eval_interval: int = 100,
        randomize_order: bool = True,
    ):
        """Initialize the progressive forgetting curriculum.

        Args:
            task_sets: Dictionary mapping task set names to task dictionaries
            env_overrides: Environment overrides
            performance_threshold: Performance threshold to trigger task switching
            smoothing: Smoothing factor for performance tracking
            switch_interval: Minimum steps between task switches
            eval_interval: Steps between evaluations of all task sets
            randomize_order: Whether to randomize the order of task sets
        """
        # Flatten all tasks for the base curriculum
        all_tasks = {}
        self.task_set_mapping = {}
        for task_set_name, tasks in task_sets.items():
            for task_name, weight in tasks.items():
                all_tasks[task_name] = weight
                self.task_set_mapping[task_name] = task_set_name

        super().__init__(all_tasks, env_overrides)

        self.task_sets = task_sets
        self.performance_threshold = performance_threshold
        self.smoothing = smoothing
        self.switch_interval = switch_interval
        self.eval_interval = eval_interval
        self.randomize_order = randomize_order

        # Performance tracking
        self.task_set_performance = {name: 0.0 for name in task_sets.keys()}
        self.current_task_set = None
        self.steps_since_switch = 0
        self.steps_since_eval = 0
        self.task_set_order = list(task_sets.keys())

        if randomize_order:
            random.shuffle(self.task_set_order)

        # Initialize with first task set
        if self.task_set_order:
            self.current_task_set = self.task_set_order[0]
            self._update_task_weights()

        logger.info(f"Initialized ProgressiveForgettingCurriculum with {len(task_sets)} task sets")
        logger.info(f"Task set order: {self.task_set_order}")

    def _update_task_weights(self):
        """Update task weights to focus on current task set."""
        if self.current_task_set is None:
            return

        # Set all tasks to zero weight
        for task_name in self._task_weights:
            self._task_weights[task_name] = 0.0

        # Set current task set tasks to equal weight
        current_tasks = self.task_sets[self.current_task_set]
        weight_per_task = 1.0 / len(current_tasks)
        for task_name in current_tasks:
            if task_name in self._task_weights:
                self._task_weights[task_name] = weight_per_task

        logger.debug(f"Switched to task set: {self.current_task_set}")

    def _evaluate_all_task_sets(self):
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
                    self.smoothing * avg_performance + (1 - self.smoothing) * old_perf
                )

        logger.info(f"Task set performances: {self.task_set_performance}")

    def _should_switch_task_set(self) -> bool:
        """Determine if we should switch to the next task set."""
        if self.current_task_set is None:
            return False

        current_perf = self.task_set_performance[self.current_task_set]
        return current_perf >= self.performance_threshold and self.steps_since_switch >= self.switch_interval

    def _switch_to_next_task_set(self):
        """Switch to the next task set in the order."""
        if self.current_task_set is None:
            return

        current_idx = self.task_set_order.index(self.current_task_set)
        next_idx = (current_idx + 1) % len(self.task_set_order)
        self.current_task_set = self.task_set_order[next_idx]

        self.steps_since_switch = 0
        self._update_task_weights()

        logger.info(f"Switched to task set: {self.current_task_set}")

    def complete_task(self, id: str, score: float):
        """Complete a task and update curriculum state."""
        super().complete_task(id, score)

        self.steps_since_switch += 1
        self.steps_since_eval += 1

        # Evaluate all task sets periodically
        if self.steps_since_eval >= self.eval_interval:
            self._evaluate_all_task_sets()
            self.steps_since_eval = 0

        # Check if we should switch task sets
        if self._should_switch_task_set():
            self._switch_to_next_task_set()

    def get_curriculum_stats(self) -> Dict[str, float]:
        """Return curriculum statistics for logging."""
        stats = {
            "current_task_set": self.current_task_set or "none",
            "steps_since_switch": self.steps_since_switch,
            "steps_since_eval": self.steps_since_eval,
        }

        # Add performance stats for each task set
        for task_set_name, performance in self.task_set_performance.items():
            stats[f"perf_{task_set_name}"] = performance

        return stats
