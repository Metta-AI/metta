"""
Bucket analysis component for curriculum systems.

Handles bucket value extraction, completion density tracking, and bucket statistics
without mixing in task management or learning progress calculations.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np


def _make_default_dict_int():
    """Factory function for creating defaultdict(int) - needed for pickling."""
    return defaultdict(int)


def _make_deque_maxlen_100():
    return deque(maxlen=100)


class BucketAnalyzer:
    """Analyzes task completion patterns across different bucket dimensions."""

    def __init__(self, max_bucket_axes: int = 3, logging_detailed_slices: bool = False):
        self.max_bucket_axes = max_bucket_axes
        self.logging_detailed_slices = logging_detailed_slices

        # Bucket tracking: bucket_name -> task_id -> value
        self._bucket_tracking: Dict[str, Dict[int, Any]] = defaultdict(dict)

        # Completion counts per bucket bin: bucket_name -> bin_index -> count
        self._bucket_completion_counts: Dict[str, Dict[int, int]] = defaultdict(_make_default_dict_int)

        # Bucket binning configuration: bucket_name -> bin_edges
        self._bucket_bins: Dict[str, List[float]] = {}

        # Track if bucket contains discrete vs continuous values
        self._bucket_is_discrete: Dict[str, bool] = {}

        # Recent completion history for density analysis
        self._bucket_completion_history: Dict[str, deque] = defaultdict(_make_deque_maxlen_100)

        # Monitored buckets (limited by max_bucket_axes)
        self._monitored_buckets: set = set()

        # Cache for expensive density statistics
        self._density_stats_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._density_cache_valid = False

    def extract_bucket_values(self, task) -> Dict[str, Any]:
        """Extract bucket values from a task's environment configuration."""
        bucket_values = {}

        # This is a placeholder - real implementation would extract from task.get_env_cfg()
        # and match against known bucket paths like "game.map_builder.width"
        if hasattr(task, "get_bucket_values"):
            bucket_values = task.get_bucket_values()

        return bucket_values

    def track_task_completion(self, task_id: int, bucket_values: Dict[str, Any], score: float) -> None:
        """Track a task completion with its bucket values and score."""
        if not bucket_values:
            return

        # Update tracking for each bucket dimension
        for bucket_name, bucket_value in bucket_values.items():
            # Only monitor if we haven't exceeded max axes
            if bucket_name not in self._monitored_buckets and len(self._monitored_buckets) >= self.max_bucket_axes:
                continue

            self._monitored_buckets.add(bucket_name)
            self._bucket_tracking[bucket_name][task_id] = bucket_value

            # Set up binning if not already configured
            if bucket_name not in self._bucket_bins:
                self._setup_bucket_binning(bucket_name, bucket_value)

            # Track completion in appropriate bin
            bin_index = self._get_bucket_bin_index(bucket_name, bucket_value)
            if bin_index is not None:
                self._bucket_completion_counts[bucket_name][bin_index] += 1
                self._bucket_completion_history[bucket_name].append((bin_index, score))

        # Invalidate density cache when new data is added
        self._density_cache_valid = False

    def _setup_bucket_binning(self, bucket_name: str, sample_value: Any) -> None:
        """Set up binning configuration for a bucket based on its value type."""
        if isinstance(sample_value, (int, float)):
            # For numeric values, assume continuous and set up 10 bins initially
            # In a real implementation, you'd analyze value distributions
            self._bucket_is_discrete[bucket_name] = False
            if isinstance(sample_value, int) and sample_value < 20:
                # Small integers are probably discrete
                self._bucket_is_discrete[bucket_name] = True
                self._bucket_bins[bucket_name] = list(range(0, 21))
            else:
                # Continuous values get adaptive binning
                self._bucket_bins[bucket_name] = [float(i) for i in range(11)]  # 0-10 initial range
        else:
            # Non-numeric values are discrete
            self._bucket_is_discrete[bucket_name] = True
            self._bucket_bins[bucket_name] = [str(sample_value)]

    def _get_bucket_bin_index(self, bucket_name: str, value: Any) -> Optional[int]:
        """Get the bin index for a value in the specified bucket."""
        if bucket_name not in self._bucket_bins:
            return None

        bins = self._bucket_bins[bucket_name]
        is_discrete = self._bucket_is_discrete.get(bucket_name, False)

        if is_discrete:
            # For discrete values, find exact match or add new bin
            if value in bins:
                return bins.index(value)
            else:
                # Add new discrete value
                bins.append(value)
                return len(bins) - 1
        else:
            # For continuous values, use numpy digitize
            if isinstance(value, (int, float)):
                return int(np.digitize([value], bins)[0])
            return None

    def get_completion_density_stats(self) -> Dict[str, Dict[str, float]]:
        """Get completion density statistics for all monitored buckets."""
        if self._density_cache_valid and self._density_stats_cache is not None:
            return self._density_stats_cache

        stats = {}

        for bucket_name in self._monitored_buckets:
            bucket_stats = self._compute_bucket_density_stats(bucket_name)
            if bucket_stats:
                stats[bucket_name] = bucket_stats

        # Cache the results
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

    def get_global_stats(self) -> Dict[str, float]:
        """Get global bucket analysis statistics."""
        stats = {
            "num_monitored_buckets": float(len(self._monitored_buckets)),
            "max_bucket_axes": float(self.max_bucket_axes),
        }

        # Add summary statistics across all buckets
        total_tasks = 0
        total_bins = 0

        for bucket_name in self._monitored_buckets:
            if bucket_name in self._bucket_tracking:
                total_tasks += len(self._bucket_tracking[bucket_name])
            if bucket_name in self._bucket_completion_counts:
                total_bins += len(self._bucket_completion_counts[bucket_name])

        stats["total_tracked_tasks"] = float(total_tasks)
        stats["total_bucket_bins"] = float(total_bins)
        stats["avg_bins_per_bucket"] = (
            float(total_bins / len(self._monitored_buckets)) if self._monitored_buckets else 0.0
        )

        return stats

    def remove_task(self, task_id: int) -> None:
        """Remove a task from all bucket tracking."""
        for bucket_name in list(self._bucket_tracking.keys()):
            self._bucket_tracking[bucket_name].pop(task_id, None)

        # Invalidate density cache when task data changes
        self._density_cache_valid = False
