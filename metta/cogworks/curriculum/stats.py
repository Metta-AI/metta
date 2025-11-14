"""
Unified statistics logging system for curriculum components.

Provides StatsLogger base class for consistent statistics interfaces.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .lp_scorers import LPScorer
    from .task_tracker import TaskTracker


class StatsLogger(ABC):
    """Base class for curriculum statistics logging.

    Provides consistent interface for all curriculum components to report
    statistics with caching and prefixing.
    """

    def __init__(self):
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

    @abstractmethod
    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms should provide."""
        pass

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed statistics (expensive operations).

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

        # Add detailed stats
        detailed = self.get_detailed_stats()
        stats.update(detailed)

        # Add prefix to all keys
        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        # Cache result
        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats


class LPStatsAggregator:
    """Aggregates statistics from learning progress components.

    Centralizes stats computation from:
    - TaskTracker (task performance data)
    - LPScorer (learning progress scores)
    """

    def __init__(
        self,
        task_tracker: "TaskTracker",
        scorer: "LPScorer",
        num_tasks: int,
    ):
        """Initialize stats aggregator.

        Args:
            task_tracker: Task performance tracker
            scorer: Learning progress scorer
            num_tasks: Total number of tasks in curriculum
        """
        self.task_tracker = task_tracker
        self.scorer = scorer
        self.num_tasks = num_tasks

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics from all components."""
        stats = {
            "num_tasks": self.num_tasks,
        }

        # Add task tracker stats with prefix
        tracker_stats = self.task_tracker.get_global_stats()
        for key, value in tracker_stats.items():
            stats[f"tracker/{key}"] = value

        return stats

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed statistics from all components."""
        stats = {}

        # Learning progress stats from scorer with lp/ prefix
        lp_stats = self.scorer.get_stats()
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

        return stats

    def get_all_stats(self) -> Dict[str, float]:
        """Get all statistics (base + detailed).

        Returns:
            Dictionary of all stats
        """
        stats = self.get_base_stats()
        stats.update(self.get_detailed_stats())
        return stats


class CacheCoordinator:
    """Coordinates cache invalidation across curriculum components.

    Centralizes cache management for:
    - Algorithm stats cache
    - Scorer task score cache
    """

    def __init__(
        self,
        stats_logger: Optional[StatsLogger] = None,
        scorer: Optional["LPScorer"] = None,
    ):
        """Initialize cache coordinator.

        Args:
            stats_logger: Optional stats logger with cache
            scorer: Optional learning progress scorer with cache
        """
        self.stats_logger = stats_logger
        self.scorer = scorer

    def invalidate_all(self) -> None:
        """Invalidate all caches across all components."""
        if self.stats_logger:
            self.stats_logger.invalidate_cache()
        if self.scorer:
            self.scorer.invalidate_cache()

    def invalidate_stats_cache(self) -> None:
        """Invalidate only the stats cache."""
        if self.stats_logger:
            self.stats_logger.invalidate_cache()

    def invalidate_scorer_cache(self) -> None:
        """Invalidate only the scorer cache."""
        if self.scorer:
            self.scorer.invalidate_cache()

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
