"""
Task tracking component for curriculum algorithms.

Handles task memory, performance history, and basic task metadata
without mixing in learning progress calculations or bucket analysis.
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple


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

    def track_task_creation(self, task_id: int) -> None:
        """Track when a task is created."""
        timestamp = time.time()
        self._task_memory[task_id] = (timestamp, 0, 0.0, 0.0)
        self._task_creation_order.append((timestamp, task_id))

        # Cleanup old tasks if we exceed memory limit
        if len(self._task_memory) > self.max_memory_tasks:
            self._cleanup_old_tasks()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance with new completion score."""
        if task_id not in self._task_memory:
            self.track_task_creation(task_id)

        creation_time, completion_count, total_score, _ = self._task_memory[task_id]
        new_completion_count = completion_count + 1
        new_total_score = total_score + score

        self._task_memory[task_id] = (creation_time, new_completion_count, new_total_score, score)
        self._completion_history.append(score)

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
        self._task_memory.pop(task_id, None)
        # Note: We don't remove from creation_order for performance - cleanup handles this

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics."""
        if not self._completion_history:
            return {
                "mean_recent_score": 0.0,
                "total_tracked_tasks": 0,
                "total_completions": 0,
            }

        total_completions = sum(completion_count for _, completion_count, _, _ in self._task_memory.values())

        return {
            "mean_recent_score": sum(self._completion_history) / len(self._completion_history),
            "total_tracked_tasks": len(self._task_memory),
            "total_completions": total_completions,
        }

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks when memory limit is exceeded."""
        cleanup_count = min(100, len(self._task_memory) - self.max_memory_tasks + 100)

        # Remove oldest tasks
        removed_count = 0
        while self._task_creation_order and removed_count < cleanup_count:
            _, task_id = self._task_creation_order.popleft()
            if task_id in self._task_memory:
                del self._task_memory[task_id]
                removed_count += 1
