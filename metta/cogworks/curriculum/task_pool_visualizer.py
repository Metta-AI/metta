"""Task pool visualization for curriculum learning analysis.

This module provides functionality to track and visualize the evolution of task pools
during curriculum learning, generating histograms for scores, completions, labels,
and mean scores by label.
"""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
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

    def generate_artifact_plots(self, histograms: Dict[str, Any], epoch: int) -> List[str]:
        """Generate plot files for wandb artifacts.

        Args:
            histograms: Dictionary containing histogram data
            epoch: Current training epoch

        Returns:
            List of file paths to generated plots
        """
        plot_files = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Task scores histogram
            if "task_scores" in histograms:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    histograms["task_scores"],
                    bins=min(30, len(histograms["task_scores"]) // 2 + 1),
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_xlabel("Task Scores")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Task Scores Distribution - Epoch {epoch}")
                ax.grid(True, alpha=0.3)

                score_file = temp_path / f"task_scores_epoch_{epoch}.png"
                fig.savefig(score_file, dpi=150, bbox_inches="tight")
                plot_files.append(str(score_file))
                plt.close(fig)

            # 2. Task completions histogram
            if "task_completions" in histograms:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    histograms["task_completions"],
                    bins=min(30, len(histograms["task_completions"]) // 2 + 1),
                    alpha=0.7,
                    edgecolor="black",
                    color="orange",
                )
                ax.set_xlabel("Task Completions")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Task Completions Distribution - Epoch {epoch}")
                ax.grid(True, alpha=0.3)

                comp_file = temp_path / f"task_completions_epoch_{epoch}.png"
                fig.savefig(comp_file, dpi=150, bbox_inches="tight")
                plot_files.append(str(comp_file))
                plt.close(fig)

            # 3. Label counts bar chart
            if "label_counts" in histograms:
                labels = histograms["label_counts"]["labels"]
                counts = histograms["label_counts"]["counts"]

                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(labels)), counts, alpha=0.7, color="green")
                ax.set_xlabel("Task Labels")
                ax.set_ylabel("Number of Tasks")
                ax.set_title(f"Task Count by Label - Epoch {epoch}")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, count in zip(bars, counts, strict=False):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f"{count}", ha="center", va="bottom")

                label_file = temp_path / f"label_counts_epoch_{epoch}.png"
                fig.savefig(label_file, dpi=150, bbox_inches="tight")
                plot_files.append(str(label_file))
                plt.close(fig)

            # 4. Mean scores by label bar chart
            if "mean_scores_by_label" in histograms:
                labels = histograms["mean_scores_by_label"]["labels"]
                means = histograms["mean_scores_by_label"]["means"]

                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(labels)), means, alpha=0.7, color="purple")
                ax.set_xlabel("Task Labels")
                ax.set_ylabel("Mean Score")
                ax.set_title(f"Mean Scores by Label - Epoch {epoch}")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, mean in zip(bars, means, strict=False):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{mean:.3f}", ha="center", va="bottom")

                mean_file = temp_path / f"mean_scores_by_label_epoch_{epoch}.png"
                fig.savefig(mean_file, dpi=150, bbox_inches="tight")
                plot_files.append(str(mean_file))
                plt.close(fig)

            # Copy files to a persistent location
            import shutil

            persistent_files = []
            for file_path in plot_files:
                filename = Path(file_path).name
                persistent_path = f"/tmp/curriculum_plots_{filename}"
                shutil.copy2(file_path, persistent_path)
                persistent_files.append(persistent_path)

            return persistent_files

    def mark_logged(self, epoch: int) -> None:
        """Mark that visualization was logged for this epoch."""
        self.last_logged_epoch = epoch
