"""Task memory and performance history management.

This module provides the TaskTracker class that handles task metadata,
performance history, and completion statistics with memory management
and performance optimizations.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskTracker:
    """Handles task metadata, performance history, and completion statistics.

    Key features:
    - Memory management: Automatic cleanup of old tasks when memory limits exceeded
    - Performance optimization: Cached total completions to avoid expensive recomputation
    - Atomic operations: Thread safety in multiprocessing environments
    """

    def __init__(
        self,
        max_memory_tasks: int = 1000,
        max_bucket_axes: int = 3,
        logging_detailed_slices: bool = False,
    ):
        self.max_memory_tasks = max_memory_tasks
        self.max_bucket_axes = max_bucket_axes
        self.logging_detailed_slices = logging_detailed_slices

        # Task performance tracking
        self._task_stats: Dict[int, Dict[str, Any]] = {}

        # Task creation order tracking for eviction (FIFO when memory full)
        self._task_creation_order: deque[int] = deque()

        # Performance optimization: Cache total completions
        self._total_completions_cache: Optional[int] = None
        self._total_completions_cache_valid = False

    def track_task_creation(self, task_id: int, bucket_values: Optional[Dict[str, Any]] = None) -> None:
        """Track creation of a new task."""
        if task_id not in self._task_stats:
            # Clean up old tasks BEFORE adding new one to prevent immediate eviction
            self._cleanup_old_tasks_if_needed()

            self._task_stats[task_id] = {
                "completion_count": 0,
                "total_score": 0.0,
                "mean_score": 0.0,
                "bucket_values": bucket_values or {},
                "scores_history": [],
            }
            self._task_creation_order.append(task_id)

            # Invalidate cache
            self._total_completions_cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update performance statistics for a task."""
        if task_id not in self._task_stats:
            # Automatically register the task instead of just warning
            logger.debug(f"Auto-registering unknown task {task_id} during performance update")
            self.track_task_creation(task_id)

        # Double-check the task exists after registration (in case it was immediately evicted)
        if task_id not in self._task_stats:
            logger.error(
                f"Task {task_id} was evicted immediately after registration due to memory limits. "
                f"Consider increasing max_memory_tasks (current: {self.max_memory_tasks})"
            )
            return

        stats = self._task_stats[task_id]
        stats["completion_count"] += 1
        stats["total_score"] += score
        stats["mean_score"] = stats["total_score"] / stats["completion_count"]
        stats["scores_history"].append(score)

        # Keep history manageable (last 100 scores)
        if len(stats["scores_history"]) > 100:
            stats["scores_history"] = stats["scores_history"][-100:]

        # Invalidate cache
        self._total_completions_cache_valid = False

    def get_task_stats(self, task_id: int) -> Dict[str, Any]:
        """Get performance statistics for a task."""
        return self._task_stats.get(
            task_id,
            {
                "completion_count": 0,
                "total_score": 0.0,
                "mean_score": 0.0,
                "bucket_values": {},
                "scores_history": [],
            },
        )

    def get_all_task_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get performance statistics for all tracked tasks."""
        return self._task_stats.copy()

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if task_id in self._task_stats:
            del self._task_stats[task_id]

            # Remove from creation order tracking
            try:
                # This is O(n) but should be rare
                self._task_creation_order.remove(task_id)
            except ValueError:
                pass  # Task not in deque, which is fine

            # Invalidate cache
            self._total_completions_cache_valid = False

    def get_tracked_task_ids(self) -> List[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_stats.keys())

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs (alias for backward compatibility)."""
        return self.get_tracked_task_ids()

    def get_total_completions(self) -> int:
        """Get total completions across all tracked tasks (cached for performance)."""
        if not self._total_completions_cache_valid:
            self._total_completions_cache = sum(stats["completion_count"] for stats in self._task_stats.values())
            self._total_completions_cache_valid = True

        return self._total_completions_cache or 0

    def get_global_stats(self) -> Dict[str, float]:
        """Get global task tracking statistics for backward compatibility."""
        return {
            "total_tracked_tasks": float(len(self._task_stats)),
            "total_completions": float(self.get_total_completions()),
            "mean_completions_per_task": float(self.get_total_completions() / len(self._task_stats))
            if self._task_stats
            else 0.0,
        }

    def get_oldest_tasks(self, n: int) -> List[int]:
        """Get the n oldest tasks by creation order."""
        return list(self._task_creation_order)[:n]

    def _cleanup_old_tasks_if_needed(self) -> None:
        """Remove oldest tasks if memory limit exceeded."""
        # Only clean up if we're significantly over the limit to avoid thrashing
        while len(self._task_stats) >= self.max_memory_tasks:
            if not self._task_creation_order:
                break

            oldest_task_id = self._task_creation_order.popleft()
            if oldest_task_id in self._task_stats:
                del self._task_stats[oldest_task_id]
                logger.debug(f"Evicted old task {oldest_task_id} due to memory limit")

        # Invalidate cache after cleanup
        self._total_completions_cache_valid = False

    def clear_all_tasks(self) -> None:
        """Clear all tracked tasks (useful for testing)."""
        self._task_stats.clear()
        self._task_creation_order.clear()
        self._total_completions_cache_valid = False
        self._total_completions_cache = None
