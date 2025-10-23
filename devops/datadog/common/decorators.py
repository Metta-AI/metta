"""Decorators for metric collection and registration."""

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global metric registry
_METRIC_REGISTRY: dict[str, dict[str, Any]] = {}


def metric(
    name: str,
    type: str = "gauge",
    tags: list[str] | None = None,
    description: str | None = None,
):
    """Decorator to register a metric collection function.

    Example:
        @metric(
            name="github.prs.open",
            type="gauge",
            tags=["category:pull_requests", "repo:metta"],
            description="Currently open pull requests"
        )
        def get_open_prs() -> int:
            return len(get_pull_requests(state="open"))

    Args:
        name: Metric name (e.g., "github.prs.open")
        type: Metric type ("gauge", "count", "rate", "histogram")
        tags: List of tags to attach to metric
        description: Human-readable description

    Returns:
        Decorated function
    """

    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        # Register metric metadata
        _METRIC_REGISTRY[name] = {
            "function": func,
            "type": type,
            "tags": tags or [],
            "description": description or func.__doc__ or "",
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                value = func(*args, **kwargs)
                logger.debug(f"Collected metric {name}: {value}")
                return value
            except Exception as e:
                logger.error(f"Failed to collect metric {name}: {e}")
                # Return sensible default based on type
                if type in ("gauge", "count"):
                    return 0
                return None

        # Attach metadata to function for introspection
        wrapper.metric_name = name  # type: ignore
        wrapper.metric_type = type  # type: ignore
        wrapper.metric_tags = tags or []  # type: ignore
        wrapper.metric_description = description  # type: ignore

        return wrapper

    return decorator


def get_registered_metrics() -> dict[str, dict[str, Any]]:
    """Get all registered metrics.

    Returns:
        Dictionary mapping metric names to their metadata
    """
    return _METRIC_REGISTRY.copy()


def collect_all_metrics() -> dict[str, Any]:
    """Collect all registered metrics.

    Returns:
        Dictionary mapping metric names to their values
    """
    results = {}
    for name, metadata in _METRIC_REGISTRY.items():
        func = metadata["function"]
        try:
            value = func()
            results[name] = value
        except Exception as e:
            logger.error(f"Failed to collect {name}: {e}")
            results[name] = None

    return results


def clear_registry():
    """Clear the metric registry (useful for testing)."""
    _METRIC_REGISTRY.clear()
