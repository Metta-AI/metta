"""
Task tracking component for curriculum algorithms.

Handles task memory, performance history, and basic task metadata
without mixing in learning progress calculations or bucket analysis.
"""

import time
from collections import deque
from typing import Any, Dict, Optional, Tuple


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
                "total_tracked_tasks": 0,
                "total_completions": 0,
            }

        # Use cached total completions if valid, otherwise compute
        if not self._cache_valid:
            self._cached_total_completions = sum(
                completion_count for _, completion_count, _, _ in self._task_memory.values()
            )
            self._cache_valid = True

        return {
            "mean_recent_score": sum(self._completion_history) / len(self._completion_history),
            "total_tracked_tasks": len(self._task_memory),
            "total_completions": self._cached_total_completions,
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

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing."""
        return {
            "max_memory_tasks": self.max_memory_tasks,
            "task_memory": {
                task_id: {
                    "creation_time": creation_time,
                    "completion_count": completion_count,
                    "total_score": total_score,
                    "last_score": last_score,
                }
                for task_id, (creation_time, completion_count, total_score, last_score) in self._task_memory.items()
            },
            "task_creation_order": list(self._task_creation_order),
            "completion_history": list(self._completion_history),
            "cached_total_completions": self._cached_total_completions,
            "cache_valid": self._cache_valid,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        self.max_memory_tasks = state["max_memory_tasks"]

        # Restore task memory
        self._task_memory.clear()
        for task_id, task_data in state["task_memory"].items():
            self._task_memory[task_id] = (
                task_data["creation_time"],
                task_data["completion_count"],
                task_data["total_score"],
                task_data["last_score"],
            )

        # Restore creation order
        self._task_creation_order = deque(state["task_creation_order"])

        # Restore completion history
        self._completion_history = deque(state["completion_history"], maxlen=1000)

        # Restore cache state
        self._cached_total_completions = state["cached_total_completions"]
        self._cache_valid = state["cache_valid"]
