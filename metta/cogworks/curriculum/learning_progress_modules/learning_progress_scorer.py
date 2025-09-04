"""
Learning progress scoring component.

Focuses solely on calculating learning progress scores from task performance data,
without mixing in task management or bucket analysis.
"""

from typing import Dict, List, Optional

import numpy as np

from .task_tracker import TaskTracker


class LearningProgressScorer:
    """Calculates learning progress scores for tasks based on performance trends."""

    def __init__(self, ema_timescale: float = 0.001, exploration_bonus: float = 0.1):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus

        # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
        self._task_emas: Dict[int, tuple[float, float, int]] = {}

        # Cache for learning progress scores to avoid recomputation
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def update_task_ema(self, task_id: int, score: float) -> None:
        """Update EMA tracking for a task with new score."""
        if task_id not in self._task_emas:
            self._task_emas[task_id] = (score, score * score, 1)
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Update EMAs
            alpha = min(1.0, self.ema_timescale * num_samples)
            new_ema_score = (1 - alpha) * ema_score + alpha * score
            new_ema_squared = (1 - alpha) * ema_squared + alpha * (score * score)

            self._task_emas[task_id] = (new_ema_score, new_ema_squared, num_samples + 1)

        # Invalidate cache for this task when EMA is updated
        self._cache_valid_tasks.discard(task_id)

    def get_learning_progress_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            score = self.exploration_bonus
        elif task_id not in self._task_emas:
            # Tasks without EMA tracking get exploration bonus (they're new to the scorer)
            score = self.exploration_bonus
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Calculate variance from EMA
            variance = max(0.0, ema_squared - ema_score * ema_score)
            std_dev = np.sqrt(variance)

            # Learning progress is approximated by variance in performance
            # High variance = actively learning, low variance = plateaued
            learning_progress = std_dev

            # Add exploration bonus for tasks with few samples
            if num_samples < 10:
                learning_progress += self.exploration_bonus * (10 - num_samples) / 10

            score = learning_progress

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def score_tasks(self, task_ids: List[int], task_tracker: TaskTracker) -> Dict[int, float]:
        """Score all provided tasks for selection probability."""
        scores = {}

        for task_id in task_ids:
            scores[task_id] = self.get_learning_progress_score(task_id, task_tracker)

        return scores

    def recommend_eviction(self, task_ids: List[int], task_tracker: TaskTracker) -> Optional[int]:
        """Recommend which task to evict based on learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids, task_tracker)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def remove_task(self, task_id: int) -> None:
        """Remove task from tracking and clear its cache."""
        self._task_emas.pop(task_id, None)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about learning progress scoring."""
        if not self._task_emas:
            return {
                "num_tracked_tasks": 0,
                "mean_num_samples": 0.0,
                "mean_ema_score": 0.0,
                "mean_learning_progress": 0.0,
            }

        num_samples_list = [num_samples for _, _, num_samples in self._task_emas.values()]
        ema_scores = [ema_score for ema_score, _, _ in self._task_emas.values()]

        # Calculate mean learning progress from EMA data
        learning_progress_scores = []
        for ema_score, ema_squared, _num_samples in self._task_emas.values():
            variance = max(0.0, ema_squared - ema_score * ema_score)
            std_dev = np.sqrt(variance)
            learning_progress_scores.append(std_dev)

        return {
            "num_tracked_tasks": len(self._task_emas),
            "mean_num_samples": np.mean(num_samples_list),
            "mean_ema_score": np.mean(ema_scores),
            "mean_learning_progress": np.mean(learning_progress_scores) if learning_progress_scores else 0.0,
        }
