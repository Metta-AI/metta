"""Task pool visualization for curriculum learning analysis.

This module provides functionality to track and visualize the evolution of task pools
during curriculum learning, generating histograms for scores, completions, labels,
and mean scores by label.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TaskPoolVisualizer:
    """Visualizes task pool distributions during curriculum learning.

    Collects and formats data for histogram visualization of:
    - Task scores distribution
    - Task completions distribution
    - Task label counts
    - Mean scores by label
    """

    def __init__(self, epoch_interval: int = 1, max_bins: int = 50, min_tasks_for_viz: int = 5):
        """Initialize the task pool visualizer.

        Args:
            epoch_interval: How often to generate visualization data (in epochs)
            max_bins: Maximum number of bins for histograms
            min_tasks_for_viz: Minimum number of tasks required to generate visualizations
        """
        self.epoch_interval = epoch_interval
        self.max_bins = max_bins
        self.min_tasks_for_viz = min_tasks_for_viz
        self.last_logged_epoch = -1

    def should_log(self, epoch: int) -> bool:
        """Check if visualizations should be generated for this epoch."""
        return epoch >= self.last_logged_epoch + self.epoch_interval

    def collect_task_pool_distributions(self, curriculum_algorithm) -> Dict[str, Any]:
        """Collect task pool distribution data for visualization.

        Args:
            curriculum_algorithm: The learning progress algorithm instance

        Returns:
            Dictionary containing histogram data for wandb logging
        """
        try:
            # Get all tracked tasks
            tracked_tasks = curriculum_algorithm.task_tracker.get_all_tracked_tasks()

            if len(tracked_tasks) < self.min_tasks_for_viz:
                logger.info(
                    f"Insufficient tasks ({len(tracked_tasks)}) for visualization, need {self.min_tasks_for_viz}"
                )
                return {}

            # Collect task data
            task_data = self._collect_task_data(curriculum_algorithm, tracked_tasks)

            if not task_data["has_data"]:
                logger.debug("No task data available for visualization")
                return {}

            # Generate histograms
            histograms = self._generate_histograms(task_data)

            if histograms:
                logger.info(f"Generated {len(histograms)} histogram types: {list(histograms.keys())}")

            return histograms

        except Exception as e:
            logger.warning(f"Failed to collect task pool distributions: {e}")
            return {}

    def _collect_task_data(self, curriculum_algorithm, tracked_tasks: List[int]) -> Dict[str, Any]:
        """Collect raw data from tracked tasks."""
        scores = []
        completions = []
        labels = []
        scores_by_label = defaultdict(list)

        # Get task scores
        try:
            task_scores = curriculum_algorithm.score_tasks(tracked_tasks)
        except Exception as e:
            logger.warning(f"Failed to get task scores: {e}")
            task_scores = {}

        for task_id in tracked_tasks:
            # Get task statistics
            task_stats = curriculum_algorithm.task_tracker.get_task_stats(task_id)
            if task_stats is None:
                continue

            # Collect score data
            score = task_scores.get(task_id, 0.0)
            scores.append(score)

            # Collect completion data
            completion_count = task_stats.get("completion_count", 0)
            completions.append(completion_count)

            # Try to get task label from curriculum
            task_label = self._get_task_label(curriculum_algorithm, task_id)
            if task_label:
                labels.append(task_label)
                # Track scores by label for mean calculation
                if completion_count > 0:  # Only include tasks with completions for mean scores
                    mean_score = task_stats.get("mean_score", 0.0)
                    scores_by_label[task_label].append(mean_score)

        return {
            "scores": scores,
            "completions": completions,
            "labels": labels,
            "scores_by_label": scores_by_label,
            "has_data": len(scores) > 0,
        }

    def _get_task_label(self, curriculum_algorithm, task_id: int) -> Optional[str]:
        """Try to get task label from curriculum system."""
        try:
            # Try to access curriculum through the algorithm
            curriculum = getattr(curriculum_algorithm, "_curriculum", None)
            if curriculum is None:
                return f"task_{task_id}"

            # Try to get task from curriculum
            task_pool = getattr(curriculum, "_task_pool", {})
            if task_id in task_pool:
                task = task_pool[task_id]
                env_cfg = task.get_env_cfg()
                if hasattr(env_cfg, "label"):
                    return env_cfg.label

            return f"task_{task_id}"
        except Exception:
            return f"task_{task_id}"

    def _generate_histograms(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate histogram data for wandb logging."""
        histograms = {}

        # Task scores histogram
        if task_data["scores"]:
            histograms["task_scores"] = np.array(task_data["scores"])

        # Task completions histogram
        if task_data["completions"]:
            histograms["task_completions"] = np.array(task_data["completions"])

        # Label count distribution
        if task_data["labels"]:
            label_counts = defaultdict(int)
            for label in task_data["labels"]:
                label_counts[label] += 1

            if label_counts:
                # Convert to arrays for histogram
                labels = list(label_counts.keys())
                counts = list(label_counts.values())
                histograms["label_counts"] = {"labels": labels, "counts": np.array(counts)}

        # Mean scores by label
        if task_data["scores_by_label"]:
            mean_scores_by_label = {}
            for label, scores in task_data["scores_by_label"].items():
                if scores:  # Only include labels with data
                    mean_scores_by_label[label] = np.mean(scores)

            if mean_scores_by_label:
                labels = list(mean_scores_by_label.keys())
                means = list(mean_scores_by_label.values())
                histograms["mean_scores_by_label"] = {"labels": labels, "means": np.array(means)}

        return histograms

    def mark_logged(self, epoch: int) -> None:
        """Mark that visualization was logged for this epoch."""
        self.last_logged_epoch = epoch
