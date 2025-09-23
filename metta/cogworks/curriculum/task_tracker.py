"""
Task tracking component for curriculum algorithms.

Handles task memory, performance history, and basic task metadata
without mixing in learning progress calculations or bucket analysis.
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


class TaskTracker:
    """Tracks task metadata, performance history, and completion statistics."""

    def __init__(self, max_memory_tasks: int = 1000):
        self.max_memory_tasks = max_memory_tasks

        # Task memory: task_id -> (creation_time, completion_count, total_score, last_score)
        self._task_memory: Dict[int, Tuple[float, int, float, float]] = {}

        # Task creation order for efficient cleanup
        self._task_creation_order = deque()  # (timestamp, task_id) pairs

        # Performance tracking
        self._completion_history = deque(maxlen=1000)  # Recent completion scores

        # Cached values to avoid expensive recomputation
        self._cached_total_completions = 0
        self._cache_valid = False

    def track_task_creation(self, task_id: int) -> None:
        """Track when a task is created."""
        timestamp = time.time()
        self._task_memory[task_id] = (timestamp, 0, 0.0, 0.0)
        self._task_creation_order.append((timestamp, task_id))

        # Cleanup old tasks if we exceed memory limit
        if len(self._task_memory) > self.max_memory_tasks:
            self._cleanup_old_tasks()

        # Invalidate cache when task structure changes
        self._cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance with new completion score."""
        # Ensure task exists in memory with atomic operation
        if task_id not in self._task_memory:
            self.track_task_creation(task_id)

        # Use get() with default to handle race conditions in multiprocessing
        task_data = self._task_memory.get(task_id)
        if task_data is None:
            # Task was removed between check and access - recreate it
            self.track_task_creation(task_id)
            task_data = self._task_memory[task_id]

        creation_time, completion_count, total_score, _ = task_data
        new_completion_count = completion_count + 1
        new_total_score = total_score + score

        self._task_memory[task_id] = (creation_time, new_completion_count, new_total_score, score)
        self._completion_history.append(score)

        # Update cached total completions incrementally
        if self._cache_valid:
            self._cached_total_completions += 1
        else:
            self._cache_valid = False

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task."""
        if task_id not in self._task_memory:
            return None

        creation_time, completion_count, total_score, last_score = self._task_memory[task_id]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "last_score": 0.0,
                "age_seconds": time.time() - creation_time,
            }

        return {
            "completion_count": completion_count,
            "mean_score": total_score / completion_count,
            "last_score": last_score,
            "age_seconds": time.time() - creation_time,
        }

    def get_all_tracked_tasks(self) -> list[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_memory.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if task_id in self._task_memory:
            # Update cached total before removal
            if self._cache_valid:
                _, completion_count, _, _ = self._task_memory[task_id]
                self._cached_total_completions -= completion_count

            self._task_memory.pop(task_id, None)
            # Note: We don't remove from creation_order for performance - cleanup handles this

            # Invalidate cache if removal makes it invalid
            if self._cache_valid:
                self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics."""
        if not self._completion_history:
            return {
                "mean_recent_score": 0.0,
                "std_recent_score": 0.0,
                "skewness_recent_score": 0.0,
                "kurtosis_recent_score": 0.0,
                "total_tracked_tasks": 0,
                "total_completions": 0,
                "mean_pool_score": 0.0,
                "std_pool_score": 0.0,
                "skew_pool_score": 0.0,
                "kurt_pool_score": 0.0,
            }

        # Use cached total completions if valid, otherwise compute
        if not self._cache_valid:
            self._cached_total_completions = sum(
                completion_count for _, completion_count, _, _ in self._task_memory.values()
            )
            self._cache_valid = True

        # Calculate recent score statistics
        recent_scores = np.array(self._completion_history)
        mean_recent = float(np.mean(recent_scores))
        std_recent = float(np.std(recent_scores)) if len(recent_scores) > 1 else 0.0
        skew_recent = float(stats.skew(recent_scores)) if len(recent_scores) > 2 else 0.0
        kurt_recent = float(stats.kurtosis(recent_scores)) if len(recent_scores) > 3 else 0.0

        # Calculate pool score statistics (all task mean scores)
        pool_scores = []
        for total_score, completion_count, _, _ in self._task_memory.values():
            if completion_count > 0:
                pool_scores.append(total_score / completion_count)

        pool_scores_array = np.array(pool_scores) if pool_scores else np.array([0.0])
        mean_pool = float(np.mean(pool_scores_array))
        std_pool = float(np.std(pool_scores_array)) if len(pool_scores_array) > 1 else 0.0
        skew_pool = float(stats.skew(pool_scores_array)) if len(pool_scores_array) > 2 else 0.0
        kurt_pool = float(stats.kurtosis(pool_scores_array)) if len(pool_scores_array) > 3 else 0.0

        return {
            "mean_recent_score": mean_recent,
            "std_recent_score": std_recent,
            "skewness_recent_score": skew_recent,
            "kurtosis_recent_score": kurt_recent,
            "total_tracked_tasks": len(self._task_memory),
            "total_completions": self._cached_total_completions,
            "mean_pool_score": mean_pool,
            "std_pool_score": std_pool,
            "skew_pool_score": skew_pool,
            "kurt_pool_score": kurt_pool,
        }

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks when memory limit is exceeded."""
        cleanup_count = min(100, len(self._task_memory) - self.max_memory_tasks + 100)

        # Remove oldest tasks
        removed_count = 0
        while self._task_creation_order and removed_count < cleanup_count:
            _, task_id = self._task_creation_order.popleft()
            if task_id in self._task_memory:
                # Update cached total before removal
                if self._cache_valid:
                    _, completion_count, _, _ = self._task_memory[task_id]
                    self._cached_total_completions -= completion_count

                del self._task_memory[task_id]
                removed_count += 1

        # Cache may still be valid after cleanup if we tracked changes
