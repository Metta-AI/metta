"""Analysis tools for progressive forgetting curriculum experiments."""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForgettingAnalyzer:
    """Analyzer for extracting forgetting metrics from progressive curriculum runs."""

    def __init__(self, wandb_logs_path: str):
        """Initialize the analyzer.

        Args:
            wandb_logs_path: Path to wandb logs or training run directory
        """
        self.wandb_logs_path = wandb_logs_path
        self.metrics_data = None

    def load_metrics(self) -> pd.DataFrame:
        """Load metrics from wandb logs."""
        # This would need to be adapted based on the actual wandb log format
        # For now, we'll assume a CSV format with timestamps and metrics
        try:
            # Try to load from wandb logs
            self.metrics_data = pd.read_csv(f"{self.wandb_logs_path}/wandb/latest-run/files/wandb-events.jsonl")
        except Exception:
            # Fallback to looking for other log formats
            logger.warning("Could not load wandb logs, metrics analysis will be limited")
            self.metrics_data = pd.DataFrame()

        return self.metrics_data

    def extract_task_set_performances(self) -> Dict[str, List[Tuple[int, float]]]:
        """Extract performance trajectories for each task set.

        Returns:
            Dictionary mapping task set names to lists of (step, performance) tuples
        """
        if self.metrics_data is None:
            self.load_metrics()

        performances = {}

        # Look for curriculum performance metrics
        for col in self.metrics_data.columns:
            if col.startswith('perf_'):
                task_set = col[5:]  # Remove 'perf_' prefix
                if 'step' in self.metrics_data.columns:
                    steps = self.metrics_data['step'].values
                    perfs = self.metrics_data[col].values
                    performances[task_set] = list(zip(steps, perfs, strict=True))

        return performances

    def calculate_forgetting_metrics(
        self,
        task_set_performances: Dict[str, List[Tuple[int, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate forgetting metrics for all task set pairs.

        Args:
            task_set_performances: Performance trajectories for each task set

        Returns:
            Dictionary mapping task set pairs to forgetting metrics
        """
        metrics = {}
        task_sets = list(task_set_performances.keys())

        for i, task_set_1 in enumerate(task_sets):
            for j, task_set_2 in enumerate(task_sets):
                if i >= j:  # Skip self-pairs and duplicate pairs
                    continue

                pair_name = f"{task_set_1}_to_{task_set_2}"
                metrics[pair_name] = self._calculate_pair_metrics(
                    task_set_performances[task_set_1],
                    task_set_performances[task_set_2]
                )

        return metrics

    def _calculate_pair_metrics(
        self,
        perf_1: List[Tuple[int, float]],
        perf_2: List[Tuple[int, float]]
    ) -> Dict[str, float]:
        """Calculate metrics for a pair of task sets.

        Args:
            perf_1: Performance trajectory for first task set
            perf_2: Performance trajectory for second task set

        Returns:
            Dictionary of forgetting metrics
        """
        if not perf_1 or not perf_2:
            return {}

        # Find the switch point (where training switches from task set 1 to 2)
        switch_step = self._find_switch_point(perf_1, perf_2)

        if switch_step is None:
            return {}

        # Extract performance before and after switch
        perf_1_before = [p for step, p in perf_1 if step < switch_step]
        perf_1_after = [p for step, p in perf_1 if step >= switch_step]
        perf_2_before = [p for step, p in perf_2 if step < switch_step]
        perf_2_after = [p for step, p in perf_2 if step >= switch_step]

        metrics = {}

        # Zero-shot transfer (performance on task set 2 before training on it)
        if perf_2_before:
            metrics['zero_shot_transfer'] = np.mean(perf_2_before)

        # Final performance on task set 1 (forgetting)
        if perf_1_after:
            metrics['final_perf_task_1'] = np.mean(perf_1_after[-10:])  # Last 10 measurements
            metrics['peak_perf_task_1'] = np.max(perf_1_before) if perf_1_before else 0.0
            metrics['forgetting_magnitude'] = metrics['peak_perf_task_1'] - metrics['final_perf_task_1']

        # Final performance on task set 2 (learning)
        if perf_2_after:
            metrics['final_perf_task_2'] = np.mean(perf_2_after[-10:])  # Last 10 measurements
            metrics['learning_magnitude'] = metrics['final_perf_task_2'] - metrics['zero_shot_transfer']

        # Learning speed (time to reach 80% of final performance)
        if perf_2_after and len(perf_2_after) > 1:
            target_perf = 0.8 * metrics['final_perf_task_2']
            learning_speed = self._calculate_learning_speed(perf_2_after, target_perf)
            metrics['learning_speed'] = learning_speed

        # Forgetting speed (time to drop to 80% of peak performance)
        if perf_1_after and len(perf_1_after) > 1:
            target_perf = 0.8 * metrics['peak_perf_task_1']
            forgetting_speed = self._calculate_forgetting_speed(perf_1_after, target_perf)
            metrics['forgetting_speed'] = forgetting_speed

        return metrics

    def _find_switch_point(
        self,
        perf_1: List[Tuple[int, float]],
        perf_2: List[Tuple[int, float]]
    ) -> Optional[int]:
        """Find the step where training switches from task set 1 to 2."""
        if not perf_1 or not perf_2:
            return None

        # Find the first step where task set 2 performance starts improving
        # This is a heuristic - in practice, we'd want to use the curriculum's switch logs
        steps_1 = [step for step, _ in perf_1]
        steps_2 = [step for step, _ in perf_2]

        # Find the earliest step where both task sets have data
        common_steps = set(steps_1) & set(steps_2)
        if not common_steps:
            return None

        return min(common_steps)

    def _calculate_learning_speed(self, performances: List[float], target: float) -> float:
        """Calculate how quickly performance reaches the target."""
        for i, perf in enumerate(performances):
            if perf >= target:
                return i  # Return number of steps to reach target
        return len(performances)  # Never reached target

    def _calculate_forgetting_speed(self, performances: List[float], target: float) -> float:
        """Calculate how quickly performance drops below the target."""
        for i, perf in enumerate(performances):
            if perf <= target:
                return i  # Return number of steps to drop below target
        return len(performances)  # Never dropped below target

    def plot_performance_trajectories(
        self,
        task_set_performances: Dict[str, List[Tuple[int, float]]],
        save_path: Optional[str] = None
    ):
        """Plot performance trajectories for all task sets."""
        plt.figure(figsize=(12, 8))

        for task_set, performances in task_set_performances.items():
            if performances:
                steps, perfs = zip(*performances, strict=True)
                plt.plot(steps, perfs, label=task_set, marker='o', markersize=2)

        plt.xlabel('Training Steps')
        plt.ylabel('Performance')
        plt.title('Task Set Performance Trajectories')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_forgetting_matrix(
        self,
        forgetting_metrics: Dict[str, Dict[str, float]],
        metric_name: str = 'forgetting_magnitude',
        save_path: Optional[str] = None
    ):
        """Plot a matrix of forgetting metrics between task set pairs."""
        task_sets = set()
        for pair_name in forgetting_metrics.keys():
            task_set_1, task_set_2 = pair_name.split('_to_')
            task_sets.add(task_set_1)
            task_sets.add(task_set_2)

        task_sets = sorted(list(task_sets))
        n = len(task_sets)
        matrix = np.zeros((n, n))

        # Fill the matrix (order matters - rows are "from", columns are "to")
        for i, task_set_1 in enumerate(task_sets):
            for j, task_set_2 in enumerate(task_sets):
                if i == j:
                    matrix[i, j] = 0.0  # No forgetting for same task set
                else:
                    pair_name = f"{task_set_1}_to_{task_set_2}"
                    if pair_name in forgetting_metrics:
                        matrix[i, j] = forgetting_metrics[pair_name].get(metric_name, 0.0)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(matrix, cmap='Reds', aspect='auto')
        plt.colorbar(im, label=metric_name.replace('_', ' ').title())

        plt.xticks(range(n), task_sets, rotation=45)
        plt.yticks(range(n), task_sets)
        plt.title(f'{metric_name.replace("_", " ").title()} Matrix (Rows: From, Columns: To)')

        # Add text annotations
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f'{matrix[i, j]:.3f}',
                        ha="center", va="center", color="black")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
