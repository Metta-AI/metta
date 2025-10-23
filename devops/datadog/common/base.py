"""Base collector class for Datadog metrics collection."""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for metric collectors.

    All collectors should inherit from this class and implement the
    collect_metrics() method.
    """

    def __init__(self, name: str):
        """Initialize the collector.

        Args:
            name: Collector name (e.g., "github", "aws", "datadog")
        """
        self.name = name
        self.metrics: dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def collect_metrics(self) -> dict[str, Any]:
        """Collect all metrics for this collector.

        Returns:
            Dictionary mapping metric keys to values.
            Values can be int, float, or None.

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement collect_metrics()")

    def collect_safe(self) -> dict[str, Any]:
        """Safely collect metrics with error handling.

        This method wraps collect_metrics() to ensure errors don't crash
        the collection process.

        Returns:
            Dictionary of collected metrics (empty if collection fails)
        """
        try:
            self.metrics = self.collect_metrics()
            self.logger.info(f"Collected {len(self.metrics)} metrics from {self.name}")
            return self.metrics
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from {self.name}: {e}", exc_info=True)
            return {}

    def get_metric(self, key: str) -> Any | None:
        """Get a specific metric value.

        Args:
            key: Metric key

        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(key)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()

    def __repr__(self) -> str:
        """Return string representation of collector."""
        return f"{self.__class__.__name__}(name='{self.name}', metrics={len(self.metrics)})"
