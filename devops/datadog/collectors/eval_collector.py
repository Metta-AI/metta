from __future__ import annotations

from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.models import MetricSample


class EvalCollector(BaseCollector):
    """
    Collector for eval metrics (local + remote) used by the infra dashboard.

    Searches S3 recursively for eval artifacts under:
    - s3://softmax-train-dir/**/eval*/
    - s3://rain-artifacts-751442549699-us-west-2/**/eval*/

    Currently no eval artifacts exist, so emits placeholder zeros and
    a data_missing sentinel metric. When eval artifacts are discovered,
    this collector will parse them to extract:
    - success: binary success signal
    - heart_delta_pct: percentage change in hearts
    - duration_minutes: runtime duration
    """

    slug = "eval"
    metric_namespace = "metta.infra.cron"
    workflow_category = "evaluation"

    S3_BUCKETS = ["softmax-train-dir", "rain-artifacts-751442549699-us-west-2"]
    EVAL_PATTERNS = ["eval", "evaluation", "eval_results"]

    def __init__(self) -> None:
        super().__init__()
        self.s3_client = boto3.client("s3")

    def collect(self) -> List[MetricSample]:
        samples: List[MetricSample] = []

        try:
            # Search S3 for eval artifacts
            eval_results = self._discover_eval_artifacts()

            if not eval_results:
                self.logger.warning("No eval artifacts found in S3")
                samples.append(self._build_data_missing_metric(value=1.0))
                # Emit placeholder zeros when data is missing
                samples.extend(self._emit_placeholder_metrics())
                return samples

            # TODO: Parse eval artifacts when they exist
            # For now, emit data_missing = 0 but still use placeholder values
            # until we implement parsing logic
            self.logger.info("Found eval artifacts but parsing not yet implemented")
            samples.append(self._build_data_missing_metric(value=0.0))
            samples.extend(self._emit_placeholder_metrics())

        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to collect eval metrics from S3: %s", exc, exc_info=True)
            samples.append(self._build_data_missing_metric(value=1.0))
            samples.extend(self._emit_placeholder_metrics())

        if not samples:
            self.logger.warning("Eval collector produced zero metrics")
        return samples

    def _discover_eval_artifacts(self) -> List[Dict[str, str]]:
        """Search S3 buckets for eval artifacts."""
        artifacts: List[Dict[str, str]] = []

        for bucket in self.S3_BUCKETS:
            try:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                # Search recursively for eval-related paths
                for pattern in self.EVAL_PATTERNS:
                    pages = paginator.paginate(Bucket=bucket, Prefix="")
                    for page in pages:
                        for obj in page.get("Contents", []):
                            key = obj["Key"]
                            if pattern in key.lower():
                                artifacts.append({"bucket": bucket, "key": key})
            except ClientError as exc:
                self.logger.warning("Failed to search bucket %s: %s", bucket, exc)

        return artifacts

    def _build_data_missing_metric(self, value: float) -> MetricSample:
        """Emit a sentinel metric indicating whether eval data is missing."""
        return self.build_sample(
            metric="eval.data_missing",
            value=value,
            workflow_name="eval_data_availability",
            task="Data availability",
            check="S3 eval artifacts found",
            condition="< 1",
            status="pass" if value < 1.0 else "fail",
        )

    def _emit_placeholder_metrics(self) -> List[MetricSample]:
        """Emit placeholder zero metrics when data is missing."""
        samples: List[MetricSample] = []
        placeholder_data = {
            "local": {"success": 0.0, "heart_delta_pct": 0.0},
            "remote": {"success": 0.0, "heart_delta_pct": 0.0, "duration_minutes": 0.0},
        }
        samples.extend(self._local_metrics(placeholder_data))
        samples.extend(self._remote_metrics(placeholder_data))
        return samples

    def _evaluate(self, comparator: str, threshold: float, value: float, warn_threshold: float | None = None) -> str:
        status = "fail"
        if comparator in (">", ">="):
            if value >= threshold:
                status = "pass"
            elif warn_threshold is not None and value >= warn_threshold:
                status = "warn"
        elif comparator in ("<", "<="):
            if value <= threshold:
                status = "pass"
            elif warn_threshold is not None and value <= warn_threshold:
                status = "warn"
        return status

    def _build(
        self,
        metric: str,
        value: float,
        workflow_name: str,
        task: str,
        check: str,
        condition: str,
        warn_threshold: float | None = None,
    ) -> MetricSample:
        comparator = condition.strip().split(" ")[0]
        threshold = float(condition.strip().split(" ")[1])
        status = self._evaluate(comparator, threshold, value, warn_threshold=warn_threshold)
        return self.build_sample(
            metric=metric,
            value=value,
            workflow_name=workflow_name,
            task=task,
            check=check,
            condition=condition,
            status=status,
        )

    def _local_metrics(self, payload: Dict) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "local_eval"

        sample = self._build(
            metric="eval.local.success",
            value=float(payload.get("success", 0.0)),
            workflow_name=wf,
            task="Local eval",
            check="Success signal",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.eval.local.success", f"Expected fully-qualified name, got {sample.name}"
        samples.append(sample)

        sample = self._build(
            metric="eval.local.heart_delta_pct",
            value=float(payload.get("heart_delta_pct", 0.0)),
            workflow_name=wf,
            task="Local eval",
            check="Heart delta pct",
            condition=">= 0",
        )
        assert sample.name == "metta.infra.cron.eval.local.heart_delta_pct", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples

    def _remote_metrics(self, payload: Dict) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "remote_eval"

        sample = self._build(
            metric="eval.remote.success",
            value=float(payload.get("success", 0.0)),
            workflow_name=wf,
            task="Remote eval",
            check="Success signal",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.eval.remote.success", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="eval.remote.heart_delta_pct",
            value=float(payload.get("heart_delta_pct", 0.0)),
            workflow_name=wf,
            task="Remote eval",
            check="Heart delta pct",
            condition=">= 0",
        )
        assert sample.name == "metta.infra.cron.eval.remote.heart_delta_pct", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="eval.remote.duration_minutes",
            value=float(payload.get("duration_minutes", 0.0)),
            workflow_name=wf,
            task="Remote eval",
            check="Runtime minutes",
            condition="<= 60",
            warn_threshold=90,
        )
        assert sample.name == "metta.infra.cron.eval.remote.duration_minutes", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples
