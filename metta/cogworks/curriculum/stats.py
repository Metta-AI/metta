"""Parameter space completion density analysis.

This module provides the BucketAnalyzer class that tracks task completion
patterns across parameter space dimensions for underexplored region identification.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BucketAnalyzer:
    """Tracks task completion patterns across parameter space dimensions.

    Key features:
    - Performance optimization: Expensive density statistics disabled by default
    - Memory efficient: Limited monitoring to max_bucket_axes buckets
    - Provides underexplored region identification
    """

    def __init__(
        self,
        max_bucket_axes: int = 3,
        logging_detailed_slices: bool = False,
    ):
        self.max_bucket_axes = max_bucket_axes
        self.logging_detailed_slices = logging_detailed_slices

        # Bucket completion tracking: bucket_key -> completion_count
        self._bucket_completions: Dict[str, int] = {}

        # Track which bucket axes we're monitoring
        self._monitored_axes: List[str] = []

    def update_bucket_completions(self, task_id: int, bucket_values: Dict[str, Any]) -> None:
        """Update bucket completion counts for a task."""
        if not bucket_values:
            return

        # Only track up to max_bucket_axes for performance
        axes_to_track = list(bucket_values.keys())[: self.max_bucket_axes]

        # Update monitored axes list
        for axis in axes_to_track:
            if axis not in self._monitored_axes:
                self._monitored_axes.append(axis)
                if len(self._monitored_axes) > self.max_bucket_axes:
                    # Remove oldest axis if we exceed limit
                    removed_axis = self._monitored_axes.pop(0)
                    # Clean up old bucket entries for removed axis
                    self._cleanup_bucket_entries(removed_axis)

        # Update completion counts for monitored axes
        for axis in axes_to_track:
            if axis in self._monitored_axes:
                bucket_key = f"{axis}:{bucket_values[axis]}"
                self._bucket_completions[bucket_key] = self._bucket_completions.get(bucket_key, 0) + 1

    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get bucket completion statistics."""
        if not self.logging_detailed_slices:
            # Return minimal stats for performance
            return {
                "total_buckets": len(self._bucket_completions),
                "monitored_axes": len(self._monitored_axes),
            }

        # Detailed statistics (expensive, only when explicitly enabled)
        stats = {
            "total_buckets": len(self._bucket_completions),
            "monitored_axes": len(self._monitored_axes),
            "axes_names": self._monitored_axes.copy(),
        }

        # Per-axis statistics
        axis_stats = {}
        for axis in self._monitored_axes:
            axis_buckets = {k: v for k, v in self._bucket_completions.items() if k.startswith(f"{axis}:")}
            if axis_buckets:
                completions = list(axis_buckets.values())
                axis_stats[axis] = {
                    "bucket_count": len(axis_buckets),
                    "total_completions": sum(completions),
                    "mean_completions": sum(completions) / len(completions),
                    "min_completions": min(completions),
                    "max_completions": max(completions),
                }

        stats["axis_stats"] = axis_stats
        return stats

    def get_underexplored_regions(self, threshold_percentile: float = 0.2) -> List[str]:
        """Get bucket keys for underexplored regions."""
        if not self._bucket_completions:
            return []

        completions = list(self._bucket_completions.values())
        if not completions:
            return []

        # Calculate threshold for underexplored regions
        completions_sorted = sorted(completions)
        threshold_idx = int(len(completions_sorted) * threshold_percentile)
        threshold = completions_sorted[threshold_idx] if threshold_idx < len(completions_sorted) else 0

        # Find buckets below threshold
        underexplored = [bucket_key for bucket_key, count in self._bucket_completions.items() if count <= threshold]

        return underexplored

    def remove_task_bucket_data(self, task_id: int, bucket_values: Dict[str, Any]) -> None:
        """Remove bucket data for an evicted task (decrease completion counts)."""
        if not bucket_values:
            return

        for axis in self._monitored_axes:
            if axis in bucket_values:
                bucket_key = f"{axis}:{bucket_values[axis]}"
                if bucket_key in self._bucket_completions:
                    self._bucket_completions[bucket_key] = max(0, self._bucket_completions[bucket_key] - 1)
                    # Remove bucket entry if count reaches zero
                    if self._bucket_completions[bucket_key] == 0:
                        del self._bucket_completions[bucket_key]

    def clear_all_data(self) -> None:
        """Clear all bucket tracking data."""
        self._bucket_completions.clear()
        self._monitored_axes.clear()

    def _cleanup_bucket_entries(self, removed_axis: str) -> None:
        """Remove bucket entries for a no-longer-monitored axis."""
        keys_to_remove = [k for k in self._bucket_completions.keys() if k.startswith(f"{removed_axis}:")]
        for key in keys_to_remove:
            del self._bucket_completions[key]

        logger.debug(f"Cleaned up {len(keys_to_remove)} bucket entries for removed axis: {removed_axis}")

    def get_monitored_axes(self) -> List[str]:
        """Get list of currently monitored axes."""
        return self._monitored_axes.copy()
