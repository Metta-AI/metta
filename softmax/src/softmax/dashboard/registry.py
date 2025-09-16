import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class MetricGoal:
    metric_key: str
    aggregate: str
    target: float
    comparison: str
    window: str
    description: str


@dataclass(frozen=True, slots=True)
class MetricCollector:
    metric_key: str
    func: Callable[..., Any]
    goal: MetricGoal
    param_names: tuple[str, ...]


_METRIC_GOAL_REGISTRY: dict[str, MetricGoal] = {}
_METRIC_COLLECTORS: dict[str, MetricCollector] = {}


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
        )
        params = tuple(inspect.signature(func).parameters.keys())
        collector = MetricCollector(
            metric_key=metric_key,
            func=func,
            goal=goal,
            param_names=params,
        )
        _METRIC_GOAL_REGISTRY[metric_key] = goal
        _METRIC_COLLECTORS[metric_key] = collector
        func.__metric_goal__ = goal  # type: ignore[attr-defined]
        return func

    return decorator


def collect_metrics(
    **extra: Any,
) -> dict[str, float]:
    importlib.import_module("softmax.dashboard.metrics")  # register metrics

    context: dict[str, Any] = {
        "branch": "main",
        "workflow_filename": "checks.yml",
        "lookback_days": 7,
    }
    context.update(extra)

    metrics: dict[str, float] = {}
    for metric_key, collector in _METRIC_COLLECTORS.items():
        kwargs = {name: context[name] for name in collector.param_names if name in context}
        value = collector.func(**kwargs)
        if value is None:
            continue
        metrics[metric_key] = float(value)

    return metrics
