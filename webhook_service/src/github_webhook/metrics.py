"""Metrics and observability for GitHub → Asana sync.

Metrics emitted:
- github_asana.tasks_created: Counter for tasks created
- github_asana.tasks_completed: Counter for tasks marked complete
- github_asana.tasks_reopened: Counter for tasks reopened
- github_asana.assign_updates: Counter for assignee updates
- github_asana.noops: Counter for no-op operations (tags: reason)
- github_asana.mapping_failures: Counter for GitHub→Asana user mapping failures
- github_asana.dead_letter.count: Counter for exhausted retries (tags: operation)
- github_asana.sync.latency_ms: Histogram for Asana API call latency (tags: operation)
- github_asana.webhook.latency_ms: Histogram for full webhook processing latency (tags: event)
"""

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
