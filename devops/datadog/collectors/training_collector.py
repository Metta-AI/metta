from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.stable_suite_fetcher import get_latest_job_states
from devops.datadog.collectors.stable_suite_mapping import is_training_job, map_job_to_workflow
from devops.datadog.collectors.stable_suite_metrics import extract_training_metrics
from devops.datadog.models import MetricSample


class TrainingCollector(BaseCollector):
    """
    Collector for training health metrics used by the infra dashboard.

    Reads job metrics from stable_suite's JobState database:
    - devops/stable/state/{version}/jobs.sqlite
    - Extracts success, hearts, sps, and shaped metrics from JobState
    - Maps jobs to workflows using stable_suite_mapping
    """

    slug = "training"
    metric_namespace = "metta.infra.cron"
    workflow_category = "training"

    def collect(self) -> List[MetricSample]:
        samples: List[MetricSample] = []

        try:
            # Get job states from stable_suite database
            job_states = get_latest_job_states(since_days=1)

            # Filter to training jobs only
            training_jobs = [js for js in job_states if is_training_job(js.name)]

            if not training_jobs:
                self.logger.warning("No training jobs found in stable_suite")
                samples.append(self._build_data_missing_metric(value=1.0))
                samples.extend(self._emit_placeholder_metrics())
                return samples

            # Emit data_missing = 0 since we found data
            samples.append(self._build_data_missing_metric(value=0.0))

            # Group jobs by workflow and extract metrics
            workflow_metrics: Dict[str, Dict[str, float]] = defaultdict(
                lambda: {"success": 0.0, "hearts": 0.0, "sps": 0.0, "shaped": 0.0}
            )

            for job_state in training_jobs:
                try:
                    workflow = map_job_to_workflow(job_state.name)
                    metrics = extract_training_metrics(job_state)

                    # Aggregate: use latest job's metrics (or could average)
                    # For now, use the most recent job's metrics
                    workflow_metrics[workflow] = metrics
                except ValueError as exc:
                    self.logger.warning("Could not map job %s to workflow: %s", job_state.name, exc)
                    continue

            # Convert to old format for compatibility with existing metric builders
            health_data: Dict[str, Dict[str, float]] = {}
            for workflow, metrics in workflow_metrics.items():
                if workflow == "multigpu_arena_basic_easy_shaped":
                    health_data["multigpu"] = metrics
                elif workflow == "multinode_learning_progress":
                    health_data["multinode"] = metrics
                elif workflow == "local_arena_basic_easy_shaped":
                    # Not available from stable_suite
                    health_data["local_arena"] = {"checkpoint1": 0.0, "checkpoint2": 0.0}
                elif workflow == "training_bugs":
                    health_data["bugs"] = {"count": 0.0}  # Not from stable_suite

            # Emit metrics for each workflow
            samples.extend(self._multigpu_metrics(health_data))
            samples.extend(self._multinode_metrics(health_data))
            samples.extend(self._local_arena_metrics(health_data))
            samples.extend(self._bugs_metrics(health_data))

        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to collect training metrics from stable_suite: %s", exc, exc_info=True)
            samples.append(self._build_data_missing_metric(value=1.0))
            samples.extend(self._emit_placeholder_metrics())

        if not samples:
            self.logger.warning("Training collector produced zero metrics")
        return samples

    def _build_data_missing_metric(self, value: float) -> MetricSample:
        """Emit a sentinel metric indicating whether training data is missing."""
        return self.build_sample(
            metric="training.data_missing",
            value=value,
            workflow_name="training_data_availability",
            task="Data availability",
            check="stable_suite data found",
            condition="< 1",
            status="pass" if value < 1.0 else "fail",
        )

    def _emit_placeholder_metrics(self) -> List[MetricSample]:
        """Emit placeholder zero metrics when data is missing."""
        samples: List[MetricSample] = []
        placeholder_data = {
            "multigpu": {"success": 0.0, "hearts": 0.0, "sps": 0.0},
            "multinode": {"success": 0.0, "hearts": 0.0, "sps": 0.0},
            "local_arena": {"checkpoint1": 0.0, "checkpoint2": 0.0},
            "bugs": {"count": 0.0},
        }
        samples.extend(self._multigpu_metrics(placeholder_data))
        samples.extend(self._multinode_metrics(placeholder_data))
        samples.extend(self._local_arena_metrics(placeholder_data))
        samples.extend(self._bugs_metrics(placeholder_data))
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

    def _multigpu_metrics(self, health_data: Dict[str, Dict]) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "multigpu_arena_basic_easy_shaped"
        payload = health_data.get("multigpu", {})

        sample = self._build(
            metric="training.multigpu.runs_success",
            value=float(payload.get("success", 0.0)),
            workflow_name=wf,
            task="Runs successfully",
            check="Binary success",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.training.multigpu.runs_success", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="training.multigpu.hearts",
            value=float(payload.get("hearts", 0.0)),
            workflow_name=wf,
            task="Hearts",
            check="Avg hearts",
            condition=">= 0.5",
        )
        assert sample.name == "metta.infra.cron.training.multigpu.hearts", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="training.multigpu.sps",
            value=float(payload.get("sps", 0.0)),
            workflow_name=wf,
            task="SPS",
            check="Steps per second",
            condition=">= 40000",
        )
        assert sample.name == "metta.infra.cron.training.multigpu.sps", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples

    def _multinode_metrics(self, health_data: Dict[str, Dict]) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "multinode_learning_progress"
        payload = health_data.get("multinode", {})

        sample = self._build(
            metric="training.multinode.runs_success",
            value=float(payload.get("success", 0.0)),
            workflow_name=wf,
            task="Runs successfully",
            check="Binary success",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.training.multinode.runs_success", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="training.multinode.hearts",
            value=float(payload.get("hearts", 0.0)),
            workflow_name=wf,
            task="Hearts",
            check="Avg hearts",
            condition=">= 0.5",
        )
        assert sample.name == "metta.infra.cron.training.multinode.hearts", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="training.multinode.shaped",
            value=float(payload.get("sps", 0.0)),
            workflow_name=wf,
            task="Shaped",
            check="Shaped SPS",
            condition=">= 40000",
        )
        assert sample.name == "metta.infra.cron.training.multinode.shaped", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples

    def _local_arena_metrics(self, health_data: Dict[str, Dict]) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "local_arena_basic_easy_shaped"
        payload = health_data.get("local_arena", {})

        sample = self._build(
            metric="training.local_arena.first_checkpoint",
            value=float(payload.get("checkpoint1", 0.0)),
            workflow_name=wf,
            task="Runs to first checkpoint",
            check="Binary success",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.training.local_arena.first_checkpoint", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)

        sample = self._build(
            metric="training.local_arena.continues",
            value=float(payload.get("checkpoint2", 0.0)),
            workflow_name=wf,
            task="Continues from checkpoint",
            check="Binary success",
            condition=">= 1",
        )
        assert sample.name == "metta.infra.cron.training.local_arena.continues", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples

    def _bugs_metrics(self, health_data: Dict[str, Dict]) -> List[MetricSample]:
        samples: List[MetricSample] = []
        wf = "training_bugs"
        payload = health_data.get("bugs", {})

        # TODO: Implement bugs collector when we have GitHub API access
        # For now, emit placeholder zero
        sample = self._build(
            metric="training.bugs.count",
            value=float(payload.get("count", 0.0)),
            workflow_name=wf,
            task="Bugs tickets",
            check="Count",
            condition="< 1",
        )
        assert sample.name == "metta.infra.cron.training.bugs.count", (
            f"Expected fully-qualified name, got {sample.name}"
        )
        samples.append(sample)
        return samples
