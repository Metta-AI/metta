from __future__ import annotations

from typing import Any, Iterable

from softmax.dashboard.registry import get_metric_goals

_DASHBOARD_COLUMNS = 2
_WIDGET_WIDTH = 12
_WIDGET_HEIGHT = 6


def _widget_layout(index: int) -> dict[str, int]:
    row = index // _DASHBOARD_COLUMNS
    col = index % _DASHBOARD_COLUMNS
    return {
        "x": col * _WIDGET_WIDTH,
        "y": row * _WIDGET_HEIGHT,
        "width": _WIDGET_WIDTH,
        "height": _WIDGET_HEIGHT,
    }


def _query_for_metric(metric_key: str) -> str:
    return f"avg:metta.{metric_key}{{source:softmax-dashboard,metric:{metric_key},branch:$branch,workflow:$workflow}}"


def _marker_value(comparison: str, target: float) -> str:
    if comparison in ("<", "<="):
        return f"y > {target}"
    if comparison in (">", ">="):
        return f"y < {target}"
    return f"y = {target}"


def build_dashboard_definition(
    *,
    title: str = "Softmax System Health",
    description: str = "Auto-generated from softmax.dashboard.metrics",
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a Datadog dashboard definition with one widget per metric."""

    widgets: list[dict[str, Any]] = []
    for index, (metric_key, goal) in enumerate(sorted(get_metric_goals().items())):
        widgets.append(
            {
                "definition": {
                    "type": "timeseries",
                    "title": metric_key,
                    "title_size": "16",
                    "title_align": "left",
                    "requests": [
                        {
                            "q": _query_for_metric(metric_key),
                            "display_type": "line",
                            "style": {"palette": "dog_classic"},
                            "on_right_yaxis": False,
                        }
                    ],
                    "markers": [
                        {
                            "display_type": "error badge",
                            "value": _marker_value(goal.comparison, goal.target),
                            "label": goal.description,
                        }
                    ],
                },
                "layout": _widget_layout(index),
            }
        )

    dashboard: dict[str, Any] = {
        "title": title,
        "description": description,
        "layout_type": "ordered",
        "widgets": widgets,
        "template_variables": [
            {"name": "branch", "prefix": "branch", "default": "main"},
            {"name": "workflow", "prefix": "workflow"},
        ],
        "notify_list": [],
    }

    if tags:
        dashboard["tags"] = list(tags)

    return dashboard
