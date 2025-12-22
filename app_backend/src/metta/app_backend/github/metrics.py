"""Metrics and observability for GitHub → Asana sync."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_metrics_logger = logging.getLogger("github_asana_metrics")
_metrics_logger.setLevel(logging.INFO)


class GitHubAsanaMetrics:
    """Metrics tracking for GitHub → Asana webhook sync."""

    @staticmethod
    def increment_counter(metric_name: str, tags: Optional[Dict[str, Any]] = None):
        """Increment a counter metric."""
        try:
            tags_str = ""
            if tags:
                tag_pairs = [f"{k}={v}" for k, v in tags.items()]
                tags_str = f" tags={','.join(tag_pairs)}"
            _metrics_logger.info(f"METRIC: {metric_name}=1{tags_str}")
        except Exception as e:
            logger.debug(f"Failed to emit metric {metric_name}: {e}")

    @staticmethod
    def record_timing(metric_name: str, duration_ms: float, tags: Optional[Dict[str, Any]] = None):
        """Record a timing/histogram metric."""
        try:
            tags_str = ""
            if tags:
                tag_pairs = [f"{k}={v}" for k, v in tags.items()]
                tags_str = f" tags={','.join(tag_pairs)}"
            _metrics_logger.info(f"METRIC: {metric_name}={duration_ms:.2f}ms{tags_str}")
        except Exception as e:
            logger.debug(f"Failed to emit timing metric {metric_name}: {e}")

    @staticmethod
    @contextmanager
    def timed(metric_name: str, tags: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            GitHubAsanaMetrics.record_timing(metric_name, duration_ms, tags)


metrics = GitHubAsanaMetrics()
