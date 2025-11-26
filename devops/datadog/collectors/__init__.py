from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseCollector

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


# Import collectors so they self-register
from .ci_collector import CICollector  # noqa: E402
from .training_health_collector import TrainingHealthCollector  # noqa: E402
from .eval_health_collector import EvalHealthCollector  # noqa: E402

register_collector(CICollector)
register_collector(TrainingHealthCollector)
register_collector(EvalHealthCollector)
