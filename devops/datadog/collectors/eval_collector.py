from __future__ import annotations

from typing import Dict, List

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.stable_suite_fetcher import get_latest_job_states
from devops.datadog.collectors.stable_suite_mapping import is_eval_job, map_job_to_workflow
from devops.datadog.collectors.stable_suite_metrics import extract_eval_metrics
from devops.datadog.models import MetricSample


class EvalCollector(BaseCollector):
    """
    Collector for eval metrics (local + remote) used by the infra dashboard.

    Reads job metrics from stable_suite's JobState database:
    - devops/stable/state/{version}/jobs.sqlite
    - Extracts success, heart_delta_pct, and duration_minutes from JobState
    - Maps jobs to workflows using stable_suite_mapping
    """

    slug = "eval"
    metric_namespace = "metta.infra.cron"
    workflow_category = "evaluation"

    def collect(self) -> List[MetricSample]:
        samples: List[MetricSample] = []

        try:
            # Get job states from stable_suite database
            job_states = get_latest_job_states(since_days=1)

            # Filter to eval jobs only
            eval_jobs = [js for js in job_states if is_eval_job(js.name)]

            if not eval_jobs:
                self.logger.warning("No eval jobs found in stable_suite")
                samples.append(self._build_data_missing_metric(value=1.0))
                samples.extend(self._emit_placeholder_metrics())
                return samples

            # Emit data_missing = 0 since we found data
            samples.append(self._build_data_missing_metric(value=0.0))

            # Extract metrics from eval jobs
            # Note: Only remote eval exists in stable_suite (no local eval)
            remote_metrics = {"success": 0.0, "heart_delta_pct": 0.0, "duration_minutes": 0.0}
            local_metrics = {"success": 0.0, "heart_delta_pct": 0.0}  # Placeholder - no local eval

            for job_state in eval_jobs:
                try:
                    workflow = map_job_to_workflow(job_state.name)
                    if workflow == "remote_eval":
                        metrics = extract_eval_metrics(job_state)
                        remote_metrics = metrics
                except ValueError as exc:
                    self.logger.warning("Could not map job %s to workflow: %s", job_state.name, exc)
                    continue

            # Emit metrics
            payload = {"local": local_metrics, "remote": remote_metrics}
            samples.extend(self._local_metrics(payload))
            samples.extend(self._remote_metrics(payload))

        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to collect eval metrics from stable_suite: %s", exc, exc_info=True)
            samples.append(self._build_data_missing_metric(value=1.0))
            samples.extend(self._emit_placeholder_metrics())

        if not samples:
            self.logger.warning("Eval collector produced zero metrics")
        return samples

    def _build_data_missing_metric(self, value: float) -> MetricSample:
        """Emit a sentinel metric indicating whether eval data is missing."""
        return self.build_sample(
            metric="eval.data_missing",
            value=value,
            workflow_name="eval_data_availability",
            task="Data availability",
            check="stable_suite eval data found",
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
        wf = "remote_eval"
        local_data = payload.get("local", {})

        sample = self._build(
            metric="eval.local.success",
            value=float(local_data.get("success", 0.0)),
            workflow_name=wf,
            task="Local eval",
            check="Success signal",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.eval.local.success", f"Expected fully-qualified name, got {sample.name}"
        samples.append(sample)

        sample = self._build(
            metric="eval.local.heart_delta_pct",
            value=float(local_data.get("heart_delta_pct", 0.0)),
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
        remote_data = payload.get("remote", {})

        sample = self._build(
            metric="eval.remote.success",
            value=float(remote_data.get("success", 0.0)),
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
            value=float(remote_data.get("heart_delta_pct", 0.0)),
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
            value=float(remote_data.get("duration_minutes", 0.0)),
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
