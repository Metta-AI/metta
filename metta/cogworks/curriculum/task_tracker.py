import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _make_default_dict_int():
    """Factory function for creating defaultdict(int) - needed for pickling."""
    return defaultdict(int)


def _make_deque_maxlen_100():
    """Factory function for creating bounded deque - needed for pickling."""
    return deque(maxlen=100)


class TaskTracker:
    """Tracks task metadata, performance history, completion statistics, and bucket analysis.

    This is a general-purpose component used by curriculum algorithms to track
    task performance over time, manage memory usage, analyze completion patterns
    across parameter dimensions, and provide statistics for logging and algorithm decision-making.
    """

    def __init__(
        self, max_memory_tasks: int = 1000, max_bucket_axes: int = 3, enable_detailed_bucket_logging: bool = False
    ):
        self.max_memory_tasks = max_memory_tasks
        self.max_bucket_axes = max_bucket_axes
        self.enable_detailed_bucket_logging = enable_detailed_bucket_logging

        self._task_memory: Dict[int, Tuple[float, int, float, float]] = {}
        self._task_creation_order = deque()
        self._completion_history = deque(maxlen=1000)

        self._cached_total_completions = 0
        self._cache_valid = False

        self._bucket_tracking: Dict[str, Dict[int, Any]] = defaultdict(dict)
        self._bucket_completion_counts: Dict[str, Dict[int, int]] = defaultdict(_make_default_dict_int)
        self._bucket_bins: Dict[str, List[float]] = {}
        self._bucket_is_discrete: Dict[str, bool] = {}
        self._bucket_completion_history: Dict[str, deque] = defaultdict(_make_deque_maxlen_100)
        self._monitored_buckets: set = set()

        self._density_stats_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._density_cache_valid = False

    def track_task_creation(self, task_id: int) -> None:
        """Track when a task is created."""
        timestamp = time.time()
        self._task_memory[task_id] = (timestamp, 0, 0.0, 0.0)
        self._task_creation_order.append((timestamp, task_id))

        if len(self._task_memory) > self.max_memory_tasks:
            self._cleanup_old_tasks()

        self._cache_valid = False

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict[str, Any] = None) -> None:
        """Update task performance statistics and optional bucket analysis."""
        if task_id not in self._task_memory:
            self.track_task_creation(task_id)

        creation_time, completion_count, total_score, _ = self._task_memory[task_id]
        new_completion_count = completion_count + 1
        new_total_score = total_score + score

        self._task_memory[task_id] = (creation_time, new_completion_count, new_total_score, score)
        self._completion_history.append(score)

        if self._cache_valid:
            self._cached_total_completions += 1
        else:
            self._cache_valid = False

        if bucket_values:
            self._update_bucket_tracking(task_id, bucket_values, score)

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

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_memory.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if task_id in self._task_memory:
            self._task_memory.pop(task_id)
            self._task_creation_order = deque((ts, tid) for ts, tid in self._task_creation_order if tid != task_id)
            self._cache_valid = False

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks to keep memory usage under control."""
        while len(self._task_memory) > self.max_memory_tasks and self._task_creation_order:
            _, old_task_id = self._task_creation_order.popleft()
            if old_task_id in self._task_memory:
                self._task_memory.pop(old_task_id)

        self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global task tracking statistics."""
        if not self._cache_valid:
            self._cached_total_completions = sum(
                completion_count for _, completion_count, _, _ in self._task_memory.values()
            )
            self._cache_valid = True

        stats = {
            "total_tracked_tasks": float(len(self._task_memory)),
            "total_completions": float(self._cached_total_completions),
            "avg_completions_per_task": (
                float(self._cached_total_completions / len(self._task_memory)) if self._task_memory else 0.0
            ),
            "recent_completion_history_size": float(len(self._completion_history)),
            "num_monitored_buckets": float(len(self._monitored_buckets)),
            "max_bucket_axes": float(self.max_bucket_axes),
        }

        total_tasks = sum(
            len(self._bucket_tracking[bucket_name])
            for bucket_name in self._monitored_buckets
            if bucket_name in self._bucket_tracking
        )

        total_bins = sum(
            len(self._bucket_completion_counts[bucket_name])
            for bucket_name in self._monitored_buckets
            if bucket_name in self._bucket_completion_counts
        )

        stats["total_tracked_tasks_buckets"] = float(total_tasks)
        stats["total_bucket_bins"] = float(total_bins)
        stats["avg_bins_per_bucket"] = (
            float(total_bins / len(self._monitored_buckets)) if self._monitored_buckets else 0.0
        )

        return stats

    def extract_bucket_values(self, task) -> Dict[str, Any]:
        """Extract bucket values from a task's environment configuration."""
        if hasattr(task, "get_bucket_values"):
            return task.get_bucket_values()
        return {}

    def _update_bucket_tracking(self, task_id: int, bucket_values: Dict[str, Any], score: float) -> None:
        """Update bucket tracking with task completion data."""
        if not bucket_values:
            return

        for bucket_name, bucket_value in bucket_values.items():
            if bucket_name not in self._monitored_buckets and len(self._monitored_buckets) >= self.max_bucket_axes:
                continue

            self._monitored_buckets.add(bucket_name)
            self._bucket_tracking[bucket_name][task_id] = bucket_value

            if bucket_name not in self._bucket_bins:
                self._setup_bucket_binning(bucket_name, bucket_value)

            bin_index = self._get_bucket_bin_index(bucket_name, bucket_value)
            if bin_index is not None:
                self._bucket_completion_counts[bucket_name][bin_index] += 1
                self._bucket_completion_history[bucket_name].append((bin_index, score))

        self._density_cache_valid = False

    def _setup_bucket_binning(self, bucket_name: str, sample_value: Any) -> None:
        """Set up binning configuration for a bucket based on its value type."""
        if isinstance(sample_value, (int, float)):
            self._bucket_is_discrete[bucket_name] = False
            if isinstance(sample_value, int) and sample_value < 20:
                self._bucket_is_discrete[bucket_name] = True
                self._bucket_bins[bucket_name] = list(range(0, 21))
            else:
                self._bucket_bins[bucket_name] = [float(i) for i in range(11)]
        else:
            self._bucket_is_discrete[bucket_name] = True
            self._bucket_bins[bucket_name] = [str(sample_value)]

    def _get_bucket_bin_index(self, bucket_name: str, value: Any) -> Optional[int]:
        """Get the bin index for a value in the specified bucket."""
        if bucket_name not in self._bucket_bins:
            return None

        bins = self._bucket_bins[bucket_name]
        is_discrete = self._bucket_is_discrete.get(bucket_name, False)

        if is_discrete:
            if value in bins:
                return bins.index(value)
            bins.append(value)
            return len(bins) - 1
        else:
            if isinstance(value, (int, float)):
                return int(np.digitize([value], bins)[0])
            return None

    def get_completion_density_stats(self) -> Dict[str, Dict[str, float]]:
        """Get completion density statistics for all monitored buckets."""
        if self.enable_detailed_bucket_logging and self._density_cache_valid and self._density_stats_cache is not None:
            return self._density_stats_cache

        stats = {}
        if self.enable_detailed_bucket_logging:
            for bucket_name in self._monitored_buckets:
                bucket_stats = self._compute_bucket_density_stats(bucket_name)
                if bucket_stats:
                    stats[bucket_name] = bucket_stats

            self._density_stats_cache = stats
            self._density_cache_valid = True

        return stats

    def _compute_bucket_density_stats(self, bucket_name: str) -> Dict[str, float]:
        """Compute density statistics for a specific bucket."""
        if bucket_name not in self._bucket_completion_counts:
            return {}

        completion_counts = self._bucket_completion_counts[bucket_name]
        if not completion_counts:
            return {}

        counts = list(completion_counts.values())
        total_completions = sum(counts)

        if total_completions == 0:
            return {}

        return {
            "total_completions": float(total_completions),
            "num_bins_used": float(len(counts)),
            "mean_completions_per_bin": float(np.mean(counts)),
            "std_completions_per_bin": float(np.std(counts)),
            "completion_entropy": self._compute_completion_entropy(counts),
        }

    def _compute_completion_entropy(self, counts: List[int]) -> float:
        """Compute entropy of completion distribution across bins."""
        total = sum(counts)
        if total == 0:
            return 0.0

        probabilities = [count / total for count in counts if count > 0]
        return float(-sum(p * np.log2(p) for p in probabilities))
