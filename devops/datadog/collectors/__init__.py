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


# TODO: unify collector discovery/registration pattern (matches Nishad's feedback)

# Import collectors so they self-register
from devops.datadog.collectors.ci_collector import CICollector  # noqa: E402
from devops.datadog.collectors.eval_collector import EvalCollector  # noqa: E402
from devops.datadog.collectors.training_collector import TrainingCollector  # noqa: E402

register_collector(CICollector)
register_collector(EvalCollector)
register_collector(TrainingCollector)

# Import stable_suite integration modules (for side effects)
from devops.datadog.collectors import (  # noqa: E402, F401
    stable_suite_mapping,
    stable_suite_metrics,
)
