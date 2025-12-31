from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MetricKind(str, Enum):
    """Supported Datadog metric intake types."""

    GAUGE = "gauge"
    COUNT = "count"
    DISTRIBUTION = "distribution"


ACCEPTANCE_METRIC_PREFIX = "metta.infra.cron.stable.acceptance"

# Valid values for acceptance criterion tags
VALID_CATEGORIES = frozenset({"training", "ci", "eval", "hygiene"})
VALID_OPERATORS = frozenset({">=", ">", "<", "<=", "==", "in"})


class MetricSample(BaseModel):
    """Structured metric ready to be serialized for Datadog."""

    name: str
    value: float
    tags: Dict[str, str] = Field(default_factory=dict)
    kind: MetricKind = MetricKind.GAUGE
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def from_criterion(
        *,
        job: str,
        category: str,
        criterion: str,
        value: float,
        target: float,
        operator: str,
        passed: bool,
        unit: str | None = None,
        base_tags: Dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> List["MetricSample"]:
        """Emit value, target, and status metrics for an acceptance criterion.

        Args:
            job: Recipe/workflow identifier (e.g., "arena_basic_easy_shaped_train_100m").
            category: Workflow type (e.g., "training", "ci", "hygiene").
            criterion: What's being measured (e.g., "overview_sps", "runs_success").
            value: The actual measured value.
            target: The threshold from code.
            operator: Comparison operator (e.g., ">=", ">", "<").
            passed: Whether the criterion passed.
            unit: Optional unit hint (e.g., "sps", "count").
            base_tags: Additional tags to include on all metrics.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            List of 3 MetricSamples: value, target, and status metrics.

        Raises:
            ValueError: If category or operator is not in the allowed set.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}, got '{category}'")
        if operator not in VALID_OPERATORS:
            raise ValueError(f"operator must be one of {sorted(VALID_OPERATORS)}, got '{operator}'")

        ts = timestamp or datetime.now(timezone.utc)
        status_str = "pass" if passed else "fail"

        tags: Dict[str, str] = {
            "job": job,
            "category": category,
            "criterion": criterion,
            "target": str(target),
            "operator": operator,
            "status": status_str,
        }
        if unit:
            tags["unit"] = unit
        if base_tags:
            tags.update(base_tags)

        return [
            # Value metric - the actual measured value
            MetricSample(
                name=f"{ACCEPTANCE_METRIC_PREFIX}.value",
                value=value,
                tags=tags,
                timestamp=ts,
            ),
            # Target metric - the threshold from code (for line overlay)
            MetricSample(
                name=f"{ACCEPTANCE_METRIC_PREFIX}.target",
                value=target,
                tags=tags,
                timestamp=ts,
            ),
            # Status metric - boolean 1=pass, 0=fail (for stepped bars)
            MetricSample(
                name=f"{ACCEPTANCE_METRIC_PREFIX}.status",
                value=1.0 if passed else 0.0,
                tags=tags,
                timestamp=ts,
            ),
        ]

    def tag_list(self) -> List[str]:
        """Convert dict tags to datadog-compatible list form."""
        return [f"{key}:{value}" for key, value in sorted(self.tags.items())]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation used for dry runs and logging."""
        data = self.model_dump(mode="json", exclude_none=False)
        # Rename 'name' to 'metric' and convert 'kind' enum to string value for backward compatibility
        return {
            "metric": data["name"],
            "value": data["value"],
            "tags": data["tags"],
            "kind": data["kind"],  # Pydantic already serializes Enum to its value in mode="json"
            "timestamp": data["timestamp"],
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
