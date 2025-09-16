import importlib
import inspect
from typing import Any, Callable

from pydantic import BaseModel


class MetricGoal(BaseModel):
    metric_key: str
    aggregate: str
    target: float
    comparison: str
    window: str
    description: str
    func: Callable[..., float | None]


_METRIC_GOAL_REGISTRY: dict[str, MetricGoal] = {}

_metrics_imported = False


def get_metric_goals() -> dict[str, MetricGoal]:
    global _metrics_imported
    if not _metrics_imported:
        importlib.import_module("softmax.dashboard.metrics")  # register metrics
        _metrics_imported = True
    return _METRIC_GOAL_REGISTRY


def metric_goal(
    *,
    metric_key: str,
    aggregate: str,
    target: float,
    comparison: str,
    window: str,
    description: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a metric goal alongside a collector helper."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        goal = MetricGoal(
            metric_key=metric_key,
            aggregate=aggregate,
            target=target,
            comparison=comparison,
            window=window,
            description=description,
            func=func,
        )
        _METRIC_GOAL_REGISTRY[metric_key] = goal
        func.__metric_goal__ = goal  # type: ignore[attr-defined]
        return func

    return decorator


def collect_metrics(
    **extra: Any,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_key, metric_goal in get_metric_goals().items():
        kwargs = {
            name: extra[name] for name, param in inspect.signature(metric_goal.func).parameters.items() if name in extra
        }
        value = metric_goal.func(**kwargs) if kwargs else metric_goal.func()
        if value is None:
            continue
        metrics[metric_key] = float(value)

    return metrics
