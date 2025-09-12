"""
Unified statistics logging system for curriculum components.

Provides StatsLogger base class for consistent statistics interfaces and
SliceAnalyzer for analyzing probability distributions across parameter slices.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np


def _make_default_dict_int():
    """Factory function for creating defaultdict(int) - needed for pickling."""
    return defaultdict(int)


def _make_deque_maxlen_100():
    """Factory function for creating deque(maxlen=100) - needed for pickling."""
    return deque(maxlen=100)


class StatsLogger(ABC):
    """Base class for curriculum statistics logging.

    Provides consistent interface for all curriculum components to report
    statistics with caching, prefixing, and detailed logging controls.
    """

    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

    @abstractmethod
    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms should provide."""
        pass

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed statistics (expensive operations).

        Only computed when enable_detailed_logging=True.
        Override in subclasses to provide detailed metrics.
        """
        return {}

    def invalidate_cache(self):
        """Invalidate the stats cache."""
        self._stats_cache_valid = False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix.

        Args:
            prefix: String to prepend to all stat keys

        Returns:
            Dictionary of statistics with prefixed keys
        """
        cache_key = prefix if prefix else "_default"

        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Get base stats (required)
        stats = self.get_base_stats()

        # Add detailed stats if enabled
        if self.enable_detailed_logging:
            detailed = self.get_detailed_stats()
            stats.update(detailed)

        # Add prefix to all keys
        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        # Cache result
        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats


class SliceAnalyzer:
    """Analyzes probability distributions across parameter slices.

    Tracks task completion patterns across different parameter dimensions
    to understand curriculum coverage and learning patterns. "Slice" refers
    to cross-sections of the parameter space being analyzed.
    """

    def __init__(self, max_slice_axes: int = 3, enable_detailed_logging: bool = False):
        self.max_slice_axes = max_slice_axes
        self.enable_detailed_logging = enable_detailed_logging

        # Slice tracking: slice_name -> task_id -> value
        self._slice_tracking: Dict[str, Dict[int, Any]] = defaultdict(dict)

        # Completion counts per slice bin: slice_name -> bin_index -> count
        self._slice_completion_counts: Dict[str, Dict[int, int]] = defaultdict(_make_default_dict_int)

        # Slice binning configuration: slice_name -> bin_edges
        self._slice_bins: Dict[str, List[float]] = {}

        # Track if slice contains discrete vs continuous values
        self._slice_is_discrete: Dict[str, bool] = {}

        # Recent completion history for density analysis
        self._slice_completion_history: Dict[str, deque] = defaultdict(_make_deque_maxlen_100)

        # Monitored slices (limited by max_slice_axes)
        self._monitored_slices: set = set()

        # Cache for expensive density statistics
        self._density_stats_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._density_cache_valid = False

    def extract_slice_values(self, task) -> Dict[str, Any]:
        """Extract slice values from a task's environment configuration."""
        slice_values = {}

        # This is a placeholder - real implementation would extract from task.get_env_cfg()
        # and match against known slice paths like "game.map_builder.width"
        if hasattr(task, "get_slice_values"):
            slice_values = task.get_slice_values()
        elif hasattr(task, "get_bucket_values"):  # Backward compatibility
            slice_values = task.get_bucket_values()

        return slice_values

    def update_task_completion(self, task_id: int, slice_values: Dict[str, Any], score: float) -> None:
        """Update slice analysis with task completion data.

        Args:
            task_id: Unique task identifier
            slice_values: Parameter slice values for this task (e.g., {"map_size": "large", "num_agents": 4})
            score: Task completion score
        """
        # Store slice values for this task
        for slice_name, value in slice_values.items():
            self._slice_tracking[slice_name][task_id] = value

            # Initialize slice if not seen before
            if slice_name not in self._slice_bins:
                self._initialize_slice_bins(slice_name, value)

            # Update monitored slices (limited by max_slice_axes)
            if len(self._monitored_slices) < self.max_slice_axes:
                self._monitored_slices.add(slice_name)
            elif slice_name not in self._monitored_slices:
                continue  # Skip non-monitored slices

            # Get bin index and update completion count
            bin_index = self._get_bin_index(slice_name, value)
            if bin_index is not None:
                self._slice_completion_counts[slice_name][bin_index] += 1
                self._slice_completion_history[slice_name].append((bin_index, score))

        # Invalidate density cache when completion data changes
        self._density_cache_valid = False

    def get_slice_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """Get probability distribution statistics across parameter slices."""
        # Return cached result if valid
        if self._density_cache_valid and self._density_stats_cache is not None:
            return self._density_stats_cache

        stats = {}

        # Get sorted slice names to ensure consistent ordering for "first three slices"
        sorted_slice_names = sorted(self._monitored_slices)

        for slice_name in sorted_slice_names:
            if slice_name not in self._slice_completion_counts:
                continue

            completion_counts = self._slice_completion_counts[slice_name]
            if not completion_counts:
                continue

            # Basic distribution statistics
            total_completions = sum(completion_counts.values())
            num_bins_used = len(completion_counts)
            num_total_bins = len(self._slice_bins.get(slice_name, []))

            # Calculate distribution metrics
            coverage = num_bins_used / max(1, num_total_bins)
            mean_completions_per_bin = total_completions / max(1, num_bins_used)

            # Calculate entropy of slice distribution (higher = more uniform coverage)
            slice_probs = [count / total_completions for count in completion_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in slice_probs if p > 0)

            # Identify underexplored regions
            completion_values = list(completion_counts.values())
            if completion_values:
                distribution_variance = np.var(completion_values)
                underexplored_bins = sum(1 for count in completion_values if count < mean_completions_per_bin * 0.5)
            else:
                distribution_variance = 0.0
                underexplored_bins = 0

            slice_stats = {
                "total_completions": total_completions,
                "coverage": coverage,
                "mean_completions_per_bin": mean_completions_per_bin,
                "entropy": entropy,
                "distribution_variance": distribution_variance,
                "underexplored_bins": underexplored_bins,
                "num_bins_used": num_bins_used,
                "num_total_bins": num_total_bins,
            }

            # Add individual slice probabilities if detailed logging is enabled and this is one of the first 3 slices
            if self.enable_detailed_logging:
                slice_index = sorted_slice_names.index(slice_name)
                if slice_index < 3:  # First three slices
                    # Calculate probability for each slice value (bin)
                    if total_completions > 0:
                        # Get all possible bins (not just used ones)
                        all_bins = self._slice_bins.get(slice_name, [])
                        for bin_idx in range(min(len(all_bins), 20)):  # Limit to first 20 slices to avoid spam
                            count = completion_counts.get(bin_idx, 0)
                            probability = count / total_completions
                            slice_stats[f"slice_{bin_idx}_probability"] = probability
                            slice_stats[f"slice_{bin_idx}_count"] = count

                            # Add bin value for context (if available)
                            if bin_idx < len(all_bins):
                                bin_value = all_bins[bin_idx]
                                if isinstance(bin_value, (int, float)):
                                    slice_stats[f"slice_{bin_idx}_value"] = float(bin_value)
                                else:
                                    # For categorical values, we can't log the string directly to wandb
                                    # So we'll create a hash or index
                                    slice_stats[f"slice_{bin_idx}_value_hash"] = hash(str(bin_value)) % 1000000

            stats[slice_name] = slice_stats

        # Cache the result
        self._density_stats_cache = stats
        self._density_cache_valid = True
        return stats

    def get_underexplored_regions(self, slice_name: str) -> List[int]:
        """Get bin indices for underexplored regions in a slice."""
        if slice_name not in self._slice_completion_counts:
            return []

        completion_counts = self._slice_completion_counts[slice_name]
        if not completion_counts:
            return []

        mean_completions = sum(completion_counts.values()) / len(completion_counts)
        threshold = mean_completions * 0.3  # Consider bins with <30% of mean as underexplored

        underexplored = []
        for bin_index, count in completion_counts.items():
            if count < threshold:
                underexplored.append(bin_index)

        return underexplored

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic slice analysis statistics."""
        total_tracked_slices = len(self._monitored_slices)
        total_tasks_tracked = len(
            set(task_id for slice_tasks in self._slice_tracking.values() for task_id in slice_tasks.keys())
        )

        return {
            "total_tracked_slices": total_tracked_slices,
            "total_tasks_tracked": total_tasks_tracked,
            "max_slice_axes": self.max_slice_axes,
        }

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed slice distribution statistics (expensive)."""
        if not self.enable_detailed_logging:
            return {}

        # Use cached distribution stats to avoid recomputation
        distribution_stats = self.get_slice_distribution_stats()
        if distribution_stats:
            avg_coverage = np.mean([stats["coverage"] for stats in distribution_stats.values()])
            avg_entropy = np.mean([stats["entropy"] for stats in distribution_stats.values()])
            avg_completions_per_bin = np.mean(
                [stats["mean_completions_per_bin"] for stats in distribution_stats.values()]
            )
        else:
            avg_coverage = 0.0
            avg_entropy = 0.0
            avg_completions_per_bin = 0.0

        detailed_stats = {
            "avg_slice_coverage": avg_coverage,
            "avg_slice_entropy": avg_entropy,
            "avg_completions_per_bin": avg_completions_per_bin,
        }

        # Add individual slice statistics with proper prefixing
        for slice_name, slice_stats in distribution_stats.items():
            for stat_name, stat_value in slice_stats.items():
                detailed_stats[f"slice_{slice_name}_{stat_name}"] = stat_value

        return detailed_stats

    def remove_task(self, task_id: int) -> None:
        """Remove task from slice tracking."""
        for slice_tasks in self._slice_tracking.values():
            slice_tasks.pop(task_id, None)

        # Invalidate density cache when tasks are removed
        self._density_cache_valid = False

    def _initialize_slice_bins(self, slice_name: str, sample_value: Any) -> None:
        """Initialize binning for a new slice based on sample value."""
        if isinstance(sample_value, (int, float)):
            # Continuous values - create bins
            if isinstance(sample_value, int) and sample_value in range(0, 20):
                # Small integer range - treat as discrete
                self._slice_bins[slice_name] = list(range(21))
                self._slice_is_discrete[slice_name] = True
            else:
                # Continuous values - create 10 bins around sample
                center = float(sample_value)
                range_size = max(abs(center), 1.0)
                bin_edges = np.linspace(center - range_size, center + range_size, 11)
                self._slice_bins[slice_name] = bin_edges.tolist()
                self._slice_is_discrete[slice_name] = False
        else:
            # Categorical/string values - treat as discrete
            self._slice_bins[slice_name] = [sample_value]
            self._slice_is_discrete[slice_name] = True

    def _get_bin_index(self, slice_name: str, value: Any) -> Optional[int]:
        """Get bin index for a value in the specified slice."""
        if slice_name not in self._slice_bins:
            return None

        bins = self._slice_bins[slice_name]

        if self._slice_is_discrete[slice_name]:
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
