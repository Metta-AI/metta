"""
Regret tracking component for curriculum algorithms.

Computes and tracks regret (difference from optimal performance) for each task.
Based on ACCEL paper: "Adversarially Compounding Complexity by Editing Levels"
Reference: https://accelagent.github.io/
"""

import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np


class RegretTracker:
    """Tracks regret (optimal_value - achieved_value) for each task."""

    def __init__(
        self,
        max_memory_tasks: int = 1000,
        optimal_value: float = 1.0,
        regret_ema_timescale: float = 0.01,
    ):
        """Initialize regret tracker.

        Args:
            max_memory_tasks: Maximum number of tasks to track in memory
            optimal_value: The optimal/maximum achievable score for tasks
            regret_ema_timescale: Timescale for exponential moving average of regret
        """
        self.max_memory_tasks = max_memory_tasks
        self.optimal_value = optimal_value
        self.regret_ema_timescale = regret_ema_timescale

        # Task regret tracking: task_id -> (creation_time, completion_count, total_regret, last_regret, ema_regret)
        self._task_regret: Dict[int, Tuple[float, int, float, float, float]] = {}

        # Task creation order for efficient cleanup
        self._task_creation_order = deque()  # (timestamp, task_id) pairs

        # Regret history for global statistics
        self._regret_history = deque(maxlen=1000)  # Recent regret values

        # Regret change tracking (for learning progress integration)
        # task_id -> (fast_ema_regret, slow_ema_regret, num_updates)
        self._regret_ema: Dict[int, Tuple[float, float, int]] = {}

        # Cached values to avoid expensive recomputation
        self._cached_mean_regret = 0.0
        self._cache_valid = False

    def compute_regret(self, score: float) -> float:
        """Compute regret from a score.

        Args:
            score: The achieved score (typically in [0, 1])

        Returns:
            Regret value (optimal_value - score)
        """
        # Regret is the gap between optimal and achieved performance
        return max(0.0, self.optimal_value - score)

    def track_task_creation(self, task_id: int) -> None:
        """Track when a task is created."""
        timestamp = time.time()
        # Initialize with 0 regret - will be updated with first actual observation
        initial_regret = 0.0
        self._task_regret[task_id] = (timestamp, 0, 0.0, initial_regret, initial_regret)

        # Initialize EMAs with 0 - will be set to first actual regret on first update
        self._regret_ema[task_id] = (0.0, 0.0, 0)

        self._task_creation_order.append((timestamp, task_id))

        # Cleanup old tasks if we exceed memory limit
        if len(self._task_regret) > self.max_memory_tasks:
            self._cleanup_old_tasks()

        # Invalidate cache when task structure changes
        self._cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task regret with new completion score.

        Args:
            task_id: Task identifier
            score: Achieved score on this task completion
        """
        # Ensure task exists in memory
        if task_id not in self._task_regret:
            self.track_task_creation(task_id)

        # Compute regret for this score
        regret = self.compute_regret(score)

        # Update regret tracking
        task_data = self._task_regret.get(task_id)
        if task_data is None:
            # Task was removed between check and access - recreate it
            self.track_task_creation(task_id)
            task_data = self._task_regret[task_id]

        creation_time, completion_count, total_regret, _, old_ema_regret = task_data
        new_completion_count = completion_count + 1
        new_total_regret = total_regret + regret

        # Update EMA regret
        if completion_count == 0:
            # First observation - initialize EMA with actual value
            new_ema_regret = regret
        else:
            alpha = min(1.0, self.regret_ema_timescale * new_completion_count)
            new_ema_regret = (1 - alpha) * old_ema_regret + alpha * regret

        self._task_regret[task_id] = (
            creation_time,
            new_completion_count,
            new_total_regret,
            regret,
            new_ema_regret,
        )
        self._regret_history.append(regret)

        # Update bidirectional EMAs for learning progress
        self._update_regret_ema(task_id, regret)

        # Invalidate cache
        self._cache_valid = False

    def _update_regret_ema(self, task_id: int, regret: float) -> None:
        """Update fast and slow EMAs of regret for learning progress tracking."""
        if task_id not in self._regret_ema:
            self._regret_ema[task_id] = (regret, regret, 1)
            return

        fast_ema, slow_ema, num_updates = self._regret_ema[task_id]

        # First update after initialization - set both to first value
        if num_updates == 0:
            self._regret_ema[task_id] = (regret, regret, 1)
            return

        # Fast EMA tracks recent regret changes
        fast_alpha = self.regret_ema_timescale
        new_fast_ema = (1 - fast_alpha) * fast_ema + fast_alpha * regret

        # Slow EMA provides baseline for comparison
        slow_alpha = self.regret_ema_timescale * 0.2  # 5x slower
        new_slow_ema = (1 - slow_alpha) * slow_ema + slow_alpha * regret

        self._regret_ema[task_id] = (new_fast_ema, new_slow_ema, num_updates + 1)

    def get_task_regret(self, task_id: int) -> Optional[float]:
        """Get current EMA regret for a task.

        Args:
            task_id: Task identifier

        Returns:
            Current regret EMA, or None if task not tracked
        """
        if task_id not in self._task_regret:
            return None

        _, _, _, _, ema_regret = self._task_regret[task_id]
        return ema_regret

    def get_regret_progress(self, task_id: int) -> Optional[float]:
        """Get regret progress (how fast regret is decreasing).

        Negative values indicate regret is decreasing (learning is happening).
        Positive values indicate regret is increasing (performance degrading).

        Args:
            task_id: Task identifier

        Returns:
            Regret change rate (fast_ema - slow_ema), or None if insufficient data
        """
        if task_id not in self._regret_ema:
            return None

        fast_ema, slow_ema, num_updates = self._regret_ema[task_id]

        if num_updates < 2:
            return None

        # Positive = regret increasing, Negative = regret decreasing (learning!)
        return fast_ema - slow_ema

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get regret statistics for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary with regret statistics, or None if task not tracked
        """
        if task_id not in self._task_regret:
            return None

        creation_time, completion_count, total_regret, last_regret, ema_regret = self._task_regret[task_id]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_regret": self.optimal_value,
                "last_regret": self.optimal_value,
                "ema_regret": ema_regret,
                "age_seconds": time.time() - creation_time,
            }

        stats = {
            "completion_count": completion_count,
            "mean_regret": total_regret / completion_count,
            "last_regret": last_regret,
            "ema_regret": ema_regret,
            "age_seconds": time.time() - creation_time,
        }

        # Add regret progress if available
        regret_progress = self.get_regret_progress(task_id)
        if regret_progress is not None:
            stats["regret_progress"] = regret_progress

        return stats

    def get_all_tracked_tasks(self) -> list[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_regret.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        self._task_regret.pop(task_id, None)
        self._regret_ema.pop(task_id, None)
        self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global regret statistics across all tracked tasks."""
        if not self._regret_history:
            return {
                "mean_recent_regret": 0.0,
                "total_tracked_tasks": 0,
                "mean_regret": 0.0,
                "median_regret": 0.0,
            }

        # Compute mean regret from all tracked tasks
        if not self._cache_valid:
            regrets = [ema_regret for _, _, _, _, ema_regret in self._task_regret.values()]
            self._cached_mean_regret = np.mean(regrets) if regrets else 0.0
            self._cache_valid = True

        return {
            "mean_recent_regret": sum(self._regret_history) / len(self._regret_history),
            "total_tracked_tasks": len(self._task_regret),
            "mean_regret": self._cached_mean_regret,
            "median_regret": np.median([ema for _, _, _, _, ema in self._task_regret.values()])
            if self._task_regret
            else 0.0,
        }

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks until we are back under the memory cap."""
        while self._task_creation_order and len(self._task_regret) > self.max_memory_tasks:
            _, task_id = self._task_creation_order.popleft()
            if task_id not in self._task_regret:
                continue

            del self._task_regret[task_id]
            self._regret_ema.pop(task_id, None)

        self._cache_valid = False

    def get_state(self) -> Dict[str, Any]:
        """Get regret tracker state for checkpointing."""
        return {
            "max_memory_tasks": self.max_memory_tasks,
            "optimal_value": self.optimal_value,
            "regret_ema_timescale": self.regret_ema_timescale,
            "task_regret": {
                task_id: {
                    "creation_time": creation_time,
                    "completion_count": completion_count,
                    "total_regret": total_regret,
                    "last_regret": last_regret,
                    "ema_regret": ema_regret,
                }
                for task_id, (
                    creation_time,
                    completion_count,
                    total_regret,
                    last_regret,
                    ema_regret,
                ) in self._task_regret.items()
            },
            "regret_ema": {
                task_id: {
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "num_updates": num_updates,
                }
                for task_id, (fast_ema, slow_ema, num_updates) in self._regret_ema.items()
            },
            "task_creation_order": list(self._task_creation_order),
            "regret_history": list(self._regret_history),
            "cached_mean_regret": self._cached_mean_regret,
            "cache_valid": self._cache_valid,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load regret tracker state from checkpoint."""
        self.max_memory_tasks = state["max_memory_tasks"]
        self.optimal_value = state["optimal_value"]
        self.regret_ema_timescale = state["regret_ema_timescale"]

        # Restore task regret
        self._task_regret.clear()
        for task_id, task_data in state["task_regret"].items():
            self._task_regret[int(task_id)] = (
                task_data["creation_time"],
                task_data["completion_count"],
                task_data["total_regret"],
                task_data["last_regret"],
                task_data["ema_regret"],
            )

        # Restore regret EMAs
        self._regret_ema.clear()
        for task_id, ema_data in state["regret_ema"].items():
            self._regret_ema[int(task_id)] = (
                ema_data["fast_ema"],
                ema_data["slow_ema"],
                ema_data["num_updates"],
            )

        # Restore creation order
        self._task_creation_order = deque(state["task_creation_order"])

        # Restore regret history
        self._regret_history = deque(state["regret_history"], maxlen=1000)

        # Restore cache state
        self._cached_mean_regret = state["cached_mean_regret"]
        self._cache_valid = state["cache_valid"]

