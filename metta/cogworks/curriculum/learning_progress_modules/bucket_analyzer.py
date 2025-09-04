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
    """Factory function for creating deque(maxlen=100) - needed for pickling."""
    return deque(maxlen=100)


class BucketAnalyzer:
    """Analyzes task completion patterns across different bucket dimensions."""

    def __init__(self, max_bucket_axes: int = 3, enable_detailed_logging: bool = False):
        self.max_bucket_axes = max_bucket_axes
        self.enable_detailed_logging = enable_detailed_logging

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
        """Track task completion and update bucket statistics."""
        # Store bucket values for this task
        for bucket_name, value in bucket_values.items():
            self._bucket_tracking[bucket_name][task_id] = value

            # Initialize bucket if not seen before
            if bucket_name not in self._bucket_bins:
                self._initialize_bucket_bins(bucket_name, value)

            # Update monitored buckets (limited by max_bucket_axes)
            if len(self._monitored_buckets) < self.max_bucket_axes:
                self._monitored_buckets.add(bucket_name)
            elif bucket_name not in self._monitored_buckets:
                continue  # Skip non-monitored buckets

            # Get bin index and update completion count
            bin_index = self._get_bin_index(bucket_name, value)
            if bin_index is not None:
                self._bucket_completion_counts[bucket_name][bin_index] += 1
                self._bucket_completion_history[bucket_name].append((bin_index, score))

        # Invalidate density cache when completion data changes
        self._density_cache_valid = False

    def get_completion_density_stats(self) -> Dict[str, Dict[str, float]]:
        """Get completion density statistics for all monitored buckets."""
        # Return cached result if valid
        if self._density_cache_valid and self._density_stats_cache is not None:
            return self._density_stats_cache

        stats = {}

        for bucket_name in self._monitored_buckets:
            if bucket_name not in self._bucket_completion_counts:
                continue

            completion_counts = self._bucket_completion_counts[bucket_name]
            if not completion_counts:
                continue

            # Basic density statistics
            total_completions = sum(completion_counts.values())
            num_bins_used = len(completion_counts)
            num_total_bins = len(self._bucket_bins.get(bucket_name, []))

            # Calculate density metrics
            density_coverage = num_bins_used / max(1, num_total_bins)
            mean_completions_per_bin = total_completions / max(1, num_bins_used)

            # Identify underexplored regions
            completion_values = list(completion_counts.values())
            if completion_values:
                density_variance = np.var(completion_values)
                underexplored_bins = sum(1 for count in completion_values if count < mean_completions_per_bin * 0.5)
            else:
                density_variance = 0.0
                underexplored_bins = 0

            stats[bucket_name] = {
                "total_completions": total_completions,
                "density_coverage": density_coverage,
                "mean_completions_per_bin": mean_completions_per_bin,
                "density_variance": density_variance,
                "underexplored_bins": underexplored_bins,
                "num_bins_used": num_bins_used,
                "num_total_bins": num_total_bins,
            }

        # Cache the result
        self._density_stats_cache = stats
        self._density_cache_valid = True
        return stats

    def get_underexplored_regions(self, bucket_name: str) -> List[int]:
        """Get bin indices for underexplored regions in a bucket."""
        if bucket_name not in self._bucket_completion_counts:
            return []

        completion_counts = self._bucket_completion_counts[bucket_name]
        if not completion_counts:
            return []

        mean_completions = sum(completion_counts.values()) / len(completion_counts)
        threshold = mean_completions * 0.3  # Consider bins with <30% of mean as underexplored

        underexplored = []
        for bin_index, count in completion_counts.items():
            if count < threshold:
                underexplored.append(bin_index)

        return underexplored

    def get_global_stats(self) -> Dict[str, float]:
        """Get global bucket analysis statistics."""
        total_tracked_buckets = len(self._monitored_buckets)
        total_tasks_tracked = len(
            set(task_id for bucket_tasks in self._bucket_tracking.values() for task_id in bucket_tasks.keys())
        )

        # Use cached density stats to avoid recomputation
        density_stats = self.get_completion_density_stats()
        if density_stats:
            avg_density_coverage = np.mean([stats["density_coverage"] for stats in density_stats.values()])
            avg_completions_per_bin = np.mean([stats["mean_completions_per_bin"] for stats in density_stats.values()])
        else:
            avg_density_coverage = 0.0
            avg_completions_per_bin = 0.0

        return {
            "total_tracked_buckets": total_tracked_buckets,
            "total_tasks_tracked": total_tasks_tracked,
            "avg_density_coverage": avg_density_coverage,
            "avg_completions_per_bin": avg_completions_per_bin,
        }

    def remove_task(self, task_id: int) -> None:
        """Remove task from bucket tracking."""
        for bucket_tasks in self._bucket_tracking.values():
            bucket_tasks.pop(task_id, None)

        # Invalidate density cache when tasks are removed
        self._density_cache_valid = False

    def _initialize_bucket_bins(self, bucket_name: str, sample_value: Any) -> None:
        """Initialize binning for a new bucket based on sample value."""
        if isinstance(sample_value, (int, float)):
            # Continuous values - create bins
            if isinstance(sample_value, int) and sample_value in range(0, 20):
                # Small integer range - treat as discrete
                self._bucket_bins[bucket_name] = list(range(21))
                self._bucket_is_discrete[bucket_name] = True
            else:
                # Continuous values - create 10 bins around sample
                center = float(sample_value)
                range_size = max(abs(center), 1.0)
                bin_edges = np.linspace(center - range_size, center + range_size, 11)
                self._bucket_bins[bucket_name] = bin_edges.tolist()
                self._bucket_is_discrete[bucket_name] = False
        else:
            # Categorical/string values - treat as discrete
            self._bucket_bins[bucket_name] = [sample_value]
            self._bucket_is_discrete[bucket_name] = True

    def _get_bin_index(self, bucket_name: str, value: Any) -> Optional[int]:
        """Get bin index for a value in the specified bucket."""
        if bucket_name not in self._bucket_bins:
            return None

        bins = self._bucket_bins[bucket_name]

        if self._bucket_is_discrete[bucket_name]:
            # Discrete values - find exact match or add new bin
            if value in bins:
                return bins.index(value)
            else:
                # Add new bin for unseen discrete value
                bins.append(value)
                return len(bins) - 1
        else:
            # Continuous values - find appropriate bin
            bin_edges = np.array(bins)
            bin_index = np.digitize(value, bin_edges) - 1
            # Clamp to valid range
            return max(0, min(bin_index, len(bin_edges) - 2))
