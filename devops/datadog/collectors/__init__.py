"""Collectors package.

Note: Training and eval collectors were removed. Datadog metrics are now emitted
directly by the stable runner workflow. This package only retains CI collector
for GitHub workflow metrics (separate from runner metrics).
"""

from __future__ import annotations

from typing import Dict, List, Type

from devops.datadog.collectors.base import BaseCollector

COLLECTOR_REGISTRY: Dict[str, Type[BaseCollector]] = {}


def register_collector(collector_cls: Type[BaseCollector]) -> Type[BaseCollector]:
    COLLECTOR_REGISTRY[collector_cls.slug] = collector_cls
    return collector_cls


def get_collector(slug: str) -> BaseCollector:
    try:
        collector_cls = COLLECTOR_REGISTRY[slug]
    except KeyError as exc:
        available = ", ".join(sorted(COLLECTOR_REGISTRY))
        raise ValueError(f"Unknown collector '{slug}'. Available: {available}") from exc
    return collector_cls()


def available_collectors() -> List[str]:
    return sorted(COLLECTOR_REGISTRY.keys())


# Import CI collector (for GitHub workflow metrics, not runner metrics)
from devops.datadog.collectors.ci_collector import CICollector  # noqa: E402

register_collector(CICollector)
