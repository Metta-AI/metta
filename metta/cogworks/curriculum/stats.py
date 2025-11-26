"""Statistics logging infrastructure for curriculum components.

This module provides the base StatsLogger class for consistent statistics reporting
across the curriculum system. All curriculum components (Curriculum, CurriculumAlgorithm,
LPScorer) inherit from StatsLogger to provide unified stats collection.

Key component:
- StatsLogger: Abstract base with caching and automatic prefix handling

Why separate file: Statistics collection is cross-cutting and needed by many components.
Centralizing it here avoids circular dependencies and provides a single source of truth
for how stats are collected, cached, and reported.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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


class NullStatsLogger(StatsLogger):
    """Minimal stats logger implementation for testing.

    Provides no-op implementations of all stats methods.
    Useful for unit tests that don't need actual stats collection.
    """

    def get_base_stats(self) -> Dict[str, float]:
        """Return empty stats dict."""
        return {}

    def get_detailed_stats(self) -> Dict[str, float]:
        """Return empty stats dict."""
        return {}
