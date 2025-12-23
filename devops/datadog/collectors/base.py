from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List

from devops.datadog.models import MetricKind, MetricSample


class BaseCollector(ABC):
    """Base class for all metric collectors."""

    slug: str = "base"
    metric_namespace: str = "metta"
    workflow_category: str = "general"
    source: str = "cron"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"devops.datadog.collectors.{self.slug}")

    @abstractmethod
    def collect(self) -> list[MetricSample]:
        """Collect metrics for Datadog ingestion."""

    def _metric_name(self, suffix: str) -> str:
        if suffix.startswith("metta."):
            return suffix
        return f"{self.metric_namespace}.{suffix}"

    def _base_tags(self) -> Dict[str, str]:
        """Return base tags required by the Datadog ingestion plan."""
        return {
            "source": self.source,
            "workflow_category": self.workflow_category,
            "service": "infra-health-dashboard",
            "env": os.environ.get("DD_ENV", "production"),
        }

    def build_criterion_samples(
        self,
        *,
        job: str,
        category: str,
        criterion: str,
        value: float,
        target: float,
        operator: str,
        passed: bool,
        unit: str | None = None,
        tags: Dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> List[MetricSample]:
        """Build criterion metrics (value, target, status) with base tags.

        This is the preferred method for emitting acceptance-style metrics.
        It emits 3 metrics that can be visualized together in Datadog:
        - .value: The actual measured value
        - .target: The threshold from code
        - .status: Boolean 1=pass, 0=fail

        Args:
            job: Job/workflow identifier (e.g., "ci_workflow", "arena_basic_easy").
            category: Workflow type (e.g., "ci", "training", "hygiene").
            criterion: What's being measured (e.g., "flaky_tests", "overview_sps").
            value: The actual measured value.
            target: The threshold from code.
            operator: Comparison operator (e.g., ">=", ">", "<").
            passed: Whether the criterion passed.
            unit: Optional unit hint (e.g., "count", "sps").
            tags: Additional tags to merge.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            List of 3 MetricSamples (value, target, status).
        """
        merged_tags = {**self._base_tags()}
        if tags:
            merged_tags.update(tags)

        return MetricSample.from_criterion(
            job=job,
            category=category,
            criterion=criterion,
            value=value,
            target=target,
            operator=operator,
            passed=passed,
            unit=unit,
            base_tags=merged_tags,
            timestamp=timestamp,
        )

    def build_sample(
        self,
        *,
        metric: str,
        value: float,
        workflow_name: str,
        task: str,
        check: str,
        condition: str,
        status: str,
        metric_kind: MetricKind = MetricKind.GAUGE,
        tags: Dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> MetricSample:
        """Helper to build MetricSample with standardized tags.

        DEPRECATED: Use build_criterion_samples() instead for new metrics.
        This method is kept for backward compatibility.
        """
        merged_tags = {
            **self._base_tags(),
            "workflow_name": workflow_name,
            "task": task,
            "check": check,
            "condition": condition,
            "status": status,
        }
        if tags:
            merged_tags.update(tags)

        return MetricSample(
            name=self._metric_name(metric),
            value=value,
            tags=merged_tags,
            kind=metric_kind,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
