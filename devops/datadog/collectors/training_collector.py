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

    Parses S3 metadata directly from s3://softmax-train-dir/.job_metadata/*/
    to infer training job health using heuristics:
    - heartbeat_file: last heartbeat timestamp
    - restart_count: instability indicator
    - termination_reason: success/failure signal

    Heuristic success logic:
    - Success if termination_reason contains "0" or "completed"
    - Failure if "1" or "error" or missing heartbeat for > 30 minutes
    - Placeholder values: hearts=1.0 if success else 0.0, sps=0 (until real source available)
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
                    health_data["local_arena"] = {"checkpoint1": 0.0, "checkpoint2": 0.0}  # Not available from stable_suite
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

    def _infer_workflow_type(self, job_name: str) -> str:
        """Infer workflow type from job name patterns."""
        job_lower = job_name.lower()
        if "multinode" in job_lower or "4x4" in job_lower or "8x" in job_lower:
            return "multinode"
        elif "multigpu" in job_lower or "arena" in job_lower:
            return "multigpu"
        elif "local" in job_lower or "1x1" in job_lower:
            return "local_arena"
        else:
            # Default to multigpu for unknown patterns
            return "multigpu"

    def _read_heartbeat(self, prefix: str) -> Optional[datetime]:
        """Read heartbeat timestamp from S3."""
        try:
            key = f"{prefix}heartbeat_file"
            response = self.s3_client.get_object(Bucket=self.S3_BUCKET, Key=key)
            content = response["Body"].read().decode("utf-8").strip()
            # Heartbeat file typically contains a timestamp
            try:
                # Try parsing as ISO format or Unix timestamp
                if content.isdigit():
                    return datetime.fromtimestamp(int(content), tz=timezone.utc)
                else:
                    # Try ISO format
                    return datetime.fromisoformat(content.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                self.logger.warning("Could not parse heartbeat timestamp: %s", content)
                return None
        except ClientError:
            return None

    def _read_restart_count(self, prefix: str) -> int:
        """Read restart count from S3."""
        try:
            key = f"{prefix}restart_count"
            response = self.s3_client.get_object(Bucket=self.S3_BUCKET, Key=key)
            content = response["Body"].read().decode("utf-8").strip()
            return int(content) if content.isdigit() else 0
        except (ClientError, ValueError):
            return 0

    def _read_termination_reason(self, prefix: str) -> Optional[str]:
        """Read termination reason from S3."""
        try:
            key = f"{prefix}termination_reason"
            response = self.s3_client.get_object(Bucket=self.S3_BUCKET, Key=key)
            return response["Body"].read().decode("utf-8").strip()
        except ClientError:
            return None

    def _infer_success(self, heartbeat_ts: Optional[datetime], termination_reason: Optional[str]) -> bool:
        """Infer job success from metadata heuristics."""
        # Check termination reason
        if termination_reason:
            reason_lower = termination_reason.lower()
            # Success indicators
            if "0" in termination_reason or "completed" in reason_lower or "success" in reason_lower:
                return True
            # Failure indicators
            if "1" in termination_reason or "error" in reason_lower or "fail" in reason_lower:
                return False

        # Check heartbeat timeout
        if heartbeat_ts:
            now = datetime.now(timezone.utc)
            age = now - heartbeat_ts
            if age > timedelta(minutes=self.HEARTBEAT_TIMEOUT_MINUTES):
                return False
            # Recent heartbeat suggests active/successful
            return True

        # No data available - assume failure for safety
        return False

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
