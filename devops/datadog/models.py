from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class MetricKind(str, Enum):
    """Supported Datadog metric intake types."""

    GAUGE = "gauge"
    COUNT = "count"
    DISTRIBUTION = "distribution"


@dataclass
class MetricSample:
    """Structured metric ready to be serialized for Datadog."""

    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    kind: MetricKind = MetricKind.GAUGE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def tag_list(self) -> List[str]:
        """Convert dict tags to datadog-compatible list form."""
        return [f"{key}:{value}" for key, value in sorted(self.tags.items())]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation used for dry runs and logging."""
        return {
            "metric": self.name,
            "value": self.value,
            "tags": self.tags,
            "kind": self.kind.value,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_series(self):  # type: ignore[override]
        """Convert to datadog_api_client MetricSeries (import lazily)."""
        from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
        from datadog_api_client.v2.model.metric_point import MetricPoint
        from datadog_api_client.v2.model.metric_series import MetricSeries

        # Map our MetricKind to Datadog's MetricIntakeType
        # Note: DISTRIBUTION may not be available in all datadog-api-client versions
        intake_type_map = {
            MetricKind.GAUGE: MetricIntakeType.GAUGE,
            MetricKind.COUNT: MetricIntakeType.COUNT,
        }
        # Fallback to GAUGE if DISTRIBUTION is not available
        if self.kind == MetricKind.DISTRIBUTION:
            if hasattr(MetricIntakeType, "DISTRIBUTION"):
                intake_type = MetricIntakeType.DISTRIBUTION
            else:
                # DISTRIBUTION not supported, use GAUGE as fallback
                intake_type = MetricIntakeType.GAUGE
        else:
            intake_type = intake_type_map[self.kind]

        return MetricSeries(
            metric=self.name,
            type=intake_type,
            points=[
                MetricPoint(
                    timestamp=int(self.timestamp.timestamp()),
                    value=self.value,
                )
            ],
            tags=self.tag_list(),
        )
