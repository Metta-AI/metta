"""
Prioritize Regressed Curriculum Algorithm for Curriculum.

This module implements the prioritize regressed algorithm as a CurriculumAlgorithm
that can be used with Curriculum nodes to prioritize tasks where current performance
has regressed relative to peak performance.
"""

import logging

import numpy as np

from metta.mettagrid.curriculum.curriculum_algorithm import CurriculumAlgorithm, CurriculumAlgorithmHypers

logger = logging.getLogger(__name__)


class PrioritizeRegressedHypers(CurriculumAlgorithmHypers):
    """Hyperparameters for PrioritizeRegressedAlgorithm."""

    moving_avg_decay_rate: float = 0.01
    min_samples_per_task: int = 5  # Minimum samples before using prioritize regressed calculation

    def algorithm_type(self) -> str:
        return "prioritize_regressed"

    def create(self, num_tasks: int) -> CurriculumAlgorithm:
        return PrioritizeRegressedAlgorithm(num_tasks, self)


class PrioritizeRegressedAlgorithm(CurriculumAlgorithm):
    """Curriculum algorithm that prioritizes tasks where performance has regressed from peak.

    This algorithm tracks both the maximum reward achieved and the moving average of rewards
    for each task. Tasks with high max/average ratios get higher weight, meaning tasks where
    we've seen good performance but are currently performing poorly get prioritized.

    Weight calculation: weight[i] = epsilon + max_reward[i] / (average_reward[i] + epsilon)

    This means:
    - Tasks with no history get epsilon weight (minimal)
    - Tasks with consistent performance get weight ≈ 1.0
    - Tasks with regression (max >> average) get higher weights
    """

    def __init__(self, num_tasks: int, hypers: PrioritizeRegressedHypers):
        """Initialize prioritize regressed algorithm.

        Args:
            num_tasks: Number of tasks this algorithm will manage
            hypers: Hyperparameters for this algorithm
        """
        super().__init__(num_tasks, hypers)
        self.moving_avg_decay_rate = hypers.moving_avg_decay_rate
        self.min_samples_per_task = hypers.min_samples_per_task
        self.reward_averages = np.zeros(num_tasks, dtype=np.float32)
        self.reward_maxes = np.zeros(num_tasks, dtype=np.float32)
        self.task_completed_count = np.zeros(num_tasks, dtype=np.int32)

        # Reference to owning Curriculum (set by Curriculum during initialization)
        self.curriculum = None

        self.epsilon = 1e-4

    def _update_weights(self, child_idx: int, score: float) -> None:
        """Update task weights based on regression from peak performance.

        Args:
            child_idx: Index of the child that completed a task
            score: Score achieved (between 0 and 1)

        Note:
            The weights array is updated in-place. The Curriculum will handle
            normalization automatically via its _update_probabilities() method.
        """
        if child_idx >= self.num_tasks or child_idx < 0:
            logger.warning(f"Invalid child_idx {child_idx} for {self.num_tasks} tasks")
            return

        # Update moving average for the completed task
        old_average = self.reward_averages[child_idx]
        self.reward_averages[child_idx] = (1 - self.moving_avg_decay_rate) * self.reward_averages[
            child_idx
        ] + self.moving_avg_decay_rate * score

        # Update maximum reward seen for this task
        self.reward_maxes[child_idx] = max(self.reward_maxes[child_idx], score)

        # Track completion count
        self.task_completed_count[child_idx] += 1

        # Debug logging with task name from context
        task_name = self.get_task_name(child_idx)
        logger.debug(
            f"Updated task {child_idx} ({task_name}): "
            f"reward mean({old_average:.3f} -> {self.reward_averages[child_idx]:.3f}), "
            f"max({self.reward_maxes[child_idx]:.3f}), "
            f"count({self.task_completed_count[child_idx]})"
        )

        # Recalculate all weights based on max/average ratios
        # First, find the max weight among well-sampled tasks
        max_weight = 1.0  # Default weight if no tasks are well-sampled yet
        for j in range(self.num_tasks):
            if self.task_completed_count[j] >= self.min_samples_per_task:
                weight = self.epsilon + self.reward_maxes[j] / (self.reward_averages[j] + self.epsilon)
                max_weight = max(max_weight, weight)

        # Now assign weights
        for i in range(self.num_tasks):
            if self.task_completed_count[i] < self.min_samples_per_task:
                # Under-sampled tasks get the max weight to ensure fair exploration
                self.weights[i] = max_weight
            else:
                # Standard prioritize regressed calculation
                self.weights[i] = self.epsilon + self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return regression statistics for logging.

        Args:
            prefix: Prefix to add to all stat keys

        Returns:
            Dictionary of statistics with optional prefix
        """
        stats = {}

        # Overall statistics
        completed_tasks = self.task_completed_count > 0
        if np.any(completed_tasks):
            stats["pr/num_completed_tasks"] = int(np.sum(completed_tasks))
            stats["pr/total_completions"] = int(np.sum(self.task_completed_count))
            stats["pr/mean_reward_average"] = float(np.mean(self.reward_averages[completed_tasks]))
            stats["pr/mean_reward_max"] = float(np.mean(self.reward_maxes[completed_tasks]))

            # Calculate regression metrics
            avg_nonzero = self.reward_averages[completed_tasks]
            max_values = self.reward_maxes[completed_tasks]
            regression_ratios = max_values / (avg_nonzero + self.epsilon)
            stats["pr/mean_regression_ratio"] = float(np.mean(regression_ratios))
            stats["pr/max_regression_ratio"] = float(np.max(regression_ratios))

            # Individual task statistics (with names from Curriculum context if available)
            for i in range(min(3, self.num_tasks)):
                if self.task_completed_count[i] > 0:
                    task_name = self.get_task_name(i)
                    task_prefix = f"pr/{task_name}"
                    stats[f"{task_prefix}/avg"] = float(self.reward_averages[i])
                    stats[f"{task_prefix}/max"] = float(self.reward_maxes[i])
                    stats[f"{task_prefix}/count"] = int(self.task_completed_count[i])
                    stats[f"{task_prefix}/regression_ratio"] = float(
                        self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)
                    )
        else:
            stats["pr/num_completed_tasks"] = 0
            stats["pr/total_completions"] = 0

        # Add prefix if provided
        if prefix:
            return {f"{prefix}{k}": v for k, v in stats.items()}
        return stats

    def get_task_regression_ratios(self) -> np.ndarray:
        """Get current regression ratios for all tasks.

        Returns:
            Array of regression ratios (max/average) for each task
        """
        ratios = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            if self.task_completed_count[i] > 0:
                ratios[i] = self.reward_maxes[i] / (self.reward_averages[i] + self.epsilon)
            else:
                ratios[i] = self.epsilon
        return ratios

    def get_task_name(self, child_idx: int) -> str:
        """Helper method to get task name from Curriculum context.

        Args:
            child_idx: Index of the child task

        Returns:
            Task name if Curriculum context is available, otherwise generic name
        """
        if self.curriculum is not None and hasattr(self.curriculum, "full_name"):
            try:
                return self.curriculum.full_name(child_idx)
            except (IndexError, AttributeError):
                pass
        return f"task_{child_idx}"

    def reset_task_stats(self, task_idx: int) -> None:
        """Reset statistics for a specific task (useful for testing).

        Args:
            task_idx: Index of task to reset
        """
        if 0 <= task_idx < self.num_tasks:
            self.reward_averages[task_idx] = 0.0
            self.reward_maxes[task_idx] = 0.0
            self.task_completed_count[task_idx] = 0
