import importlib
import logging
import typing

logger = logging.getLogger(__name__)

_METRIC_REGISTRY: dict[str, typing.Callable[..., float | None]] = {}

_metrics_imported = False


def get_system_health_metrics() -> dict[str, typing.Callable[..., float | None]]:
    global _metrics_imported
    if not _metrics_imported:
        importlib.import_module("softmax.dashboard.metrics")  # register metrics
        _metrics_imported = True
    return _METRIC_REGISTRY


def system_health_metric(
    *,
    metric_key: str,
) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]:
    """Decorator to register a metric goal alongside a collector helper."""

    def decorator(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
        _METRIC_REGISTRY[metric_key] = func
        return func

    return decorator


def collect_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_key, metric_goal in get_system_health_metrics().items():
        try:
            value = metric_goal()
        except Exception as e:
            logger.error(f"Error collecting metric {metric_key}: {e}", exc_info=True)
            continue
        if value is None:
            continue
        metrics[metric_key] = float(value)

    return metrics
