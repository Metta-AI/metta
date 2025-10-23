"""
Unified statistics logging system for curriculum components.

Provides StatsLogger base class for consistent statistics interfaces and
SliceAnalyzer for analyzing probability distributions across parameter slices.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .lp_scorers import LPScorer
    from .task_tracker import TaskTracker


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

            # Update completion history (for density analysis)
            self._slice_completion_history[slice_name].append((value, score))

        # Invalidate density cache when new data added
        self._density_cache_valid = False

    def _initialize_slice_bins(self, slice_name: str, value: Any) -> None:
        """Initialize binning strategy for a slice based on first value."""
        if isinstance(value, (int, float)):
            # Continuous value - initialize with empty bins (will be populated as we see values)
            self._slice_bins[slice_name] = []
            self._slice_is_discrete[slice_name] = False
        else:
            # Discrete value (string, etc.) - treat categorically
            self._slice_bins[slice_name] = [value]
            self._slice_is_discrete[slice_name] = True

    def get_slice_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """Get distribution statistics for each monitored slice.

        Returns:
            Dict mapping slice_name to its distribution stats (entropy, coverage, etc.)
        """
        if self._density_cache_valid and self._density_stats_cache is not None:
            return self._density_stats_cache

        slice_stats = {}

        for slice_name in self._monitored_slices:
            counts = self._slice_completion_counts[slice_name]
            if not counts:
                continue

            # Calculate distribution metrics
            total_completions = sum(counts.values())
            num_bins = len(self._slice_bins[slice_name])
            num_bins_with_data = len(counts)

            # Coverage: fraction of bins that have been seen
            coverage = num_bins_with_data / max(1, num_bins)

            # Entropy: measure of distribution uniformity
            probs = np.array(list(counts.values())) / max(1, total_completions)
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Normalize entropy by maximum possible (log of number of bins)
            max_entropy = np.log(max(1, num_bins_with_data))
            normalized_entropy = entropy / max(1e-10, max_entropy)

            slice_stats[slice_name] = {
                "coverage": coverage,
                "entropy": normalized_entropy,
                "mean_completions_per_bin": total_completions / max(1, num_bins_with_data),
                "total_completions": total_completions,
                "num_bins": num_bins,
                "num_bins_with_data": num_bins_with_data,
            }

        self._density_stats_cache = slice_stats
        self._density_cache_valid = True

        return slice_stats

    def get_slice_value_for_task(self, task_id: int, slice_name: str) -> Optional[Any]:
        """Get the slice value for a specific task and slice dimension."""
        return self._slice_tracking.get(slice_name, {}).get(task_id)

    def get_all_slice_names(self) -> List[str]:
        """Get all tracked slice names."""
        return list(self._monitored_slices)

    def invalidate_density_cache(self):
        """Invalidate the density statistics cache."""
        self._density_cache_valid = False

    def remove_task(self, task_id: int) -> None:
        """Remove a task from slice tracking to prevent memory leaks.

        Args:
            task_id: The task ID to remove from tracking
        """
        # Remove task from all slice tracking dictionaries
        for slice_name in list(self._slice_tracking.keys()):
            self._slice_tracking[slice_name].pop(task_id, None)

        # Note: We don't remove from completion counts as those are aggregated statistics
        # that should persist even after individual tasks are evicted

        # Invalidate cache since we modified tracking data
        self._density_cache_valid = False

    def get_state(self) -> Dict[str, Any]:
        """Get slice analyzer state for checkpointing."""
        return {
            "slice_tracking": {k: dict(v) for k, v in self._slice_tracking.items()},
            "slice_completion_counts": {k: dict(v) for k, v in self._slice_completion_counts.items()},
            "slice_bins": self._slice_bins.copy(),
            "slice_is_discrete": self._slice_is_discrete.copy(),
            "monitored_slices": list(self._monitored_slices),
            # Note: completion_history is not serialized (transient data)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load slice analyzer state from checkpoint."""
        self._slice_tracking = defaultdict(dict, {k: dict(v) for k, v in state.get("slice_tracking", {}).items()})
        self._slice_completion_counts = defaultdict(
            _make_default_dict_int,
            {k: defaultdict(int, v) for k, v in state.get("slice_completion_counts", {}).items()},
        )
        self._slice_bins = state.get("slice_bins", {})
        self._slice_is_discrete = state.get("slice_is_discrete", {})
        self._monitored_slices = set(state.get("monitored_slices", []))
        # Reset completion history (transient data)
        self._slice_completion_history = defaultdict(_make_deque_maxlen_100)
        # Invalidate caches
        self._density_cache_valid = False

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic slice analysis statistics."""
        total_tracked_slices = len(self._monitored_slices)

        return {
            "total_tracked_slices": total_tracked_slices,
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

        # Add per-slice stats with slice name prefix
        for slice_name, stats in distribution_stats.items():
            # Clean slice name for stat keys (replace dots/special chars)
            clean_name = slice_name.replace(".", "_").replace("/", "_")
            for key, value in stats.items():
                detailed_stats[f"slice/{clean_name}/{key}"] = value

        return detailed_stats

    def _get_bin_index(self, slice_name: str, value: Any) -> Optional[int]:
        """Get bin index for a value in the given slice."""
        if slice_name not in self._slice_bins:
            return None

        bins = self._slice_bins[slice_name]

        if self._slice_is_discrete.get(slice_name, False):
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


class LPStatsAggregator:
    """Aggregates statistics from learning progress components.

    Centralizes stats computation from:
    - TaskTracker (task performance data)
    - LPScorer (learning progress scores)
    - SliceAnalyzer (parameter distribution analysis)
    """

    def __init__(
        self,
        task_tracker: "TaskTracker",
        scorer: "LPScorer",
        slice_analyzer: SliceAnalyzer,
        num_tasks: int,
    ):
        """Initialize stats aggregator.

        Args:
            task_tracker: Task performance tracker
            scorer: Learning progress scorer
            slice_analyzer: Parameter slice analyzer
            num_tasks: Total number of tasks in curriculum
        """
        self.task_tracker = task_tracker
        self.scorer = scorer
        self.slice_analyzer = slice_analyzer
        self.num_tasks = num_tasks

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics from all components."""
        stats = {
            "num_tasks": self.num_tasks,
            **self.slice_analyzer.get_base_stats(),
        }

        # Add task tracker stats with prefix
        tracker_stats = self.task_tracker.get_global_stats()
        for key, value in tracker_stats.items():
            stats[f"tracker/{key}"] = value

        return stats

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed statistics from all components."""
        stats = {}

        # Slice analyzer detailed stats
        stats.update(self.slice_analyzer.get_detailed_stats())

        # Learning progress stats from scorer with lp/ prefix
        lp_stats = self.scorer.get_stats()
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

        return stats

    def get_all_stats(self, enable_detailed: bool = False) -> Dict[str, float]:
        """Get all statistics (base + optionally detailed).

        Args:
            enable_detailed: Whether to include detailed stats

        Returns:
            Dictionary of all stats
        """
        stats = self.get_base_stats()

        if enable_detailed:
            stats.update(self.get_detailed_stats())

        return stats


class CacheCoordinator:
    """Coordinates cache invalidation across curriculum components.

    Centralizes cache management for:
    - Algorithm stats cache
    - Scorer task score cache
    - SliceAnalyzer density cache
    """

    def __init__(
        self,
        stats_logger: Optional[StatsLogger] = None,
        scorer: Optional["LPScorer"] = None,
        slice_analyzer: Optional[SliceAnalyzer] = None,
    ):
        """Initialize cache coordinator.

        Args:
            stats_logger: Optional stats logger with cache
            scorer: Optional learning progress scorer with cache
            slice_analyzer: Optional slice analyzer with cache
        """
        self.stats_logger = stats_logger
        self.scorer = scorer
        self.slice_analyzer = slice_analyzer

    def invalidate_all(self) -> None:
        """Invalidate all caches across all components."""
        if self.stats_logger:
            self.stats_logger.invalidate_cache()
        if self.scorer:
            self.scorer.invalidate_cache()
        if self.slice_analyzer:
            self.slice_analyzer.invalidate_density_cache()

    def invalidate_stats_cache(self) -> None:
        """Invalidate only the stats cache."""
        if self.stats_logger:
            self.stats_logger.invalidate_cache()

    def invalidate_scorer_cache(self) -> None:
        """Invalidate only the scorer cache."""
        if self.scorer:
            self.scorer.invalidate_cache()

    def invalidate_slice_cache(self) -> None:
        """Invalidate only the slice analyzer cache."""
        if self.slice_analyzer:
            self.slice_analyzer.invalidate_density_cache()

    def invalidate_task(self, task_id: int) -> None:
        """Invalidate caches for a specific task.

        Args:
            task_id: Task to invalidate from caches
        """
        # Scorer has task-specific cache invalidation
        if self.scorer:
            self.scorer.remove_task(task_id)

        # Stats cache needs full invalidation when any task changes
        if self.stats_logger:
            self.stats_logger.invalidate_cache()

        # Slice cache is invalidated when new data is added, handled elsewhere
