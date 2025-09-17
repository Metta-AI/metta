"""Statistics logging and analysis for curriculum learning systems.

Provides StatsLogger for consistent statistics reporting across curriculum components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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
