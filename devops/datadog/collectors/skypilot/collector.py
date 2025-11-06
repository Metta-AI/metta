"""Skypilot metrics collector for Datadog monitoring."""

import datetime
from typing import Any

import sky
import sky.jobs

from devops.datadog.utils.base import BaseCollector


class SkypilotCollector(BaseCollector):
    """Collector for Skypilot cluster and job metrics.

    Collects comprehensive metrics about active clusters, job statuses, runtime
    distributions, resource utilization, reliability, and cost tracking from the
    Skypilot job orchestration system.

    Per-job metrics with tags (job_id, user, region, gpu_type, instance_type, status):

    Basic metrics (all jobs):
    - skypilot.job.runtime_hours: Current runtime in hours
    - skypilot.job.estimated_cost_hourly: Estimated hourly burn rate (USD)
    - skypilot.job.gpu_count: Number of GPUs allocated

    Queue health (queued jobs):
    - skypilot.job.queue_wait_seconds: Time waiting in queue

    Efficiency (running/completed jobs):
    - skypilot.job.setup_seconds: Provisioning latency (submit → start)

    Cost tracking (completed jobs):
    - skypilot.job.total_duration_hours: End-to-end duration (submit → end)
    - skypilot.job.total_cost_usd: Total job cost

    Failure analysis (failed jobs):
    - skypilot.job.time_to_failure_hours: Runtime before failure

    These tagged metrics enable dimensional analysis in Datadog (e.g., cost by user,
    SLA tracking, chargeback, failure pattern identification).
    """

    def __init__(self):
        super().__init__(name="skypilot")

    def collect_metrics(self) -> dict[str, Any]:
        metrics = {}
        metrics.update(self._collect_cluster_metrics())
        metrics.update(self._collect_job_metrics())
        return metrics

    def _collect_cluster_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            request_id = sky.status(all_users=True)
            clusters = sky.get(request_id)

            active_clusters = sum(1 for c in clusters if c.get("status") == sky.ClusterStatus.UP)
            metrics["skypilot.clusters"] = [(active_clusters, ["status:active"])]

        except Exception as e:
            self.logger.error(f"Failed to collect cluster metrics: {e}")

        return metrics

    def _collect_job_metrics(self) -> dict[str, Any]:
        metrics = {
            "skypilot.job.runtime_hours": [],
            "skypilot.job.estimated_cost_hourly": [],
            "skypilot.job.gpu_count": [],
            "skypilot.job.queue_wait_seconds": [],
            "skypilot.job.setup_seconds": [],
            "skypilot.job.total_duration_hours": [],
            "skypilot.job.total_cost_usd": [],
            "skypilot.job.time_to_failure_hours": [],
        }

        try:
            request_id = sky.jobs.queue(refresh=False, all_users=True)
            jobs = sky.get(request_id)

            now = datetime.datetime.now(datetime.timezone.utc)
            seven_days_ago = now - datetime.timedelta(days=7)

            queued_count = 0
            running_count = 0
            failed_count = 0
            failed_7d_count = 0
            succeeded_count = 0
            cancelled_count = 0

            running_durations = []
            recovery_counts = []
            active_users = set()

            runtime_bucket_0_1h = 0
            runtime_bucket_1_4h = 0
            runtime_bucket_4_24h = 0
            runtime_bucket_over_24h = 0

            gpu_counts = {"l4": 0, "a10g": 0, "h100": 0, "total": 0}
            spot_jobs = 0
            ondemand_jobs = 0
            regions = {"us_east_1": 0, "us_west_2": 0, "other": 0}
            with_recoveries = 0

            for job in jobs:
                status = job.get("status")
                user = job.get("user_name")
                if user:
                    active_users.add(user)

                if status == sky.jobs.ManagedJobStatus.PENDING:
                    queued_count += 1
                    self._emit_job_metrics(job, "queued", metrics)

                elif status == sky.jobs.ManagedJobStatus.RUNNING:
                    running_count += 1

                    duration = job.get("job_duration", 0)
                    if duration > 0:
                        running_durations.append(duration)

                        hours = duration / 3600
                        if hours < 1:
                            runtime_bucket_0_1h += 1
                        elif hours < 4:
                            runtime_bucket_1_4h += 1
                        elif hours < 24:
                            runtime_bucket_4_24h += 1
                        else:
                            runtime_bucket_over_24h += 1

                    accelerators = job.get("accelerators", {})
                    if isinstance(accelerators, dict):
                        for gpu_type, count in accelerators.items():
                            gpu_type_lower = gpu_type.lower()
                            if "l4" in gpu_type_lower:
                                gpu_counts["l4"] += count
                            elif "a10g" in gpu_type_lower:
                                gpu_counts["a10g"] += count
                            elif "h100" in gpu_type_lower:
                                gpu_counts["h100"] += count
                            gpu_counts["total"] += count

                    resources_str = job.get("cluster_resources", "")
                    if "spot" in resources_str.lower():
                        spot_jobs += 1
                    else:
                        ondemand_jobs += 1

                    region = job.get("region", "")
                    if "us-east-1" in region:
                        regions["us_east_1"] += 1
                    elif "us-west-2" in region:
                        regions["us_west_2"] += 1
                    elif region:
                        regions["other"] += 1

                    recovery_count = job.get("recovery_count", 0)
                    if recovery_count > 0:
                        with_recoveries += 1
                        recovery_counts.append(recovery_count)

                    self._emit_job_metrics(job, "running", metrics)

                elif status in (sky.jobs.ManagedJobStatus.FAILED, sky.jobs.ManagedJobStatus.FAILED_SETUP):
                    failed_count += 1

                    submitted_ts = job.get("submitted_at", 0)
                    submitted_at = datetime.datetime.fromtimestamp(submitted_ts, tz=datetime.timezone.utc)
                    if submitted_at >= seven_days_ago:
                        failed_7d_count += 1

                    self._emit_job_metrics(job, "failed", metrics)

                elif status == sky.jobs.ManagedJobStatus.SUCCEEDED:
                    succeeded_count += 1
                    self._emit_job_metrics(job, "succeeded", metrics)

                elif status == sky.jobs.ManagedJobStatus.CANCELLED:
                    cancelled_count += 1
                    self._emit_job_metrics(job, "cancelled", metrics)

            metrics["skypilot.jobs"] = [
                (queued_count, ["status:queued"]),
                (running_count, ["status:running"]),
                (failed_count, ["status:failed"]),
                (failed_7d_count, ["status:failed", "timeframe:7d"]),
                (succeeded_count, ["status:succeeded"]),
                (cancelled_count, ["status:cancelled"]),
                (runtime_bucket_0_1h, ["runtime_bucket:0_1h"]),
                (runtime_bucket_1_4h, ["runtime_bucket:1_4h"]),
                (runtime_bucket_4_24h, ["runtime_bucket:4_24h"]),
                (runtime_bucket_over_24h, ["runtime_bucket:over_24h"]),
                (spot_jobs, ["pricing:spot"]),
                (ondemand_jobs, ["pricing:ondemand"]),
                (regions["us_east_1"], ["region:us_east_1"]),
                (regions["us_west_2"], ["region:us_west_2"]),
                (regions["other"], ["region:other"]),
                (with_recoveries, ["has_recoveries:true"]),
            ]

            if running_durations:
                running_durations.sort()
                n = len(running_durations)

                metrics["skypilot.jobs.runtime_seconds"] = [
                    (running_durations[0], ["metric:min"]),
                    (running_durations[-1], ["metric:max"]),
                    (sum(running_durations) / n, ["metric:avg"]),
                    (running_durations[int(n * 0.50)], ["metric:p50"]),
                    (running_durations[int(n * 0.90)], ["metric:p90"]),
                    (running_durations[min(int(n * 0.99), n - 1)], ["metric:p99"]),
                ]

            metrics["skypilot.resources.gpus"] = [
                (gpu_counts["l4"], ["type:l4"]),
                (gpu_counts["a10g"], ["type:a10g"]),
                (gpu_counts["h100"], ["type:h100"]),
                (gpu_counts["total"], ["type:total"]),
            ]

            if recovery_counts:
                metrics["skypilot.jobs.recovery_count"] = [
                    (sum(recovery_counts) / len(recovery_counts), ["metric:avg"]),
                    (max(recovery_counts), ["metric:max"]),
                ]

            metrics["skypilot.users"] = [(len(active_users), ["status:active"])]

        except sky.exceptions.ClusterNotUpError:
            self.logger.warning("Jobs controller is not up, skipping job metrics")
        except Exception as e:
            self.logger.error(f"Failed to collect job metrics: {e}")

        return metrics

    def _emit_job_metrics(self, job: dict[str, Any], job_status: str, metrics: dict[str, Any]) -> None:
        """Emit per-job metrics with tags for dimensional analysis.

        Args:
            job: Job dictionary from SkyPilot API
            job_status: Job status ("queued", "running", etc.)
            metrics: Metrics dictionary to append to
        """
        try:
            job_id = job.get("job_id", "unknown")
            user = job.get("user_name", "unknown")
            region = job.get("region", "unknown")

            resources_str = job.get("cluster_resources", "")
            instance_type = "spot" if "spot" in resources_str.lower() else "on_demand"

            base_tags = [
                f"job_id:{job_id}",
                f"user:{user}",
                f"region:{region}",
                f"instance_type:{instance_type}",
                f"status:{job_status}",
            ]

            duration_seconds = job.get("job_duration", 0)
            if duration_seconds > 0:
                runtime_hours = duration_seconds / 3600
                metrics["skypilot.job.runtime_hours"].append((runtime_hours, base_tags.copy()))

            accelerators = job.get("accelerators", {})
            if isinstance(accelerators, dict):
                for gpu_type, count in accelerators.items():
                    gpu_type_normalized = gpu_type.lower().replace(":", "_")
                    gpu_tags = base_tags + [f"gpu_type:{gpu_type_normalized}"]

                    metrics["skypilot.job.gpu_count"].append((count, gpu_tags))

                    hourly_cost = self._estimate_gpu_cost(gpu_type, count, instance_type)
                    if hourly_cost:
                        metrics["skypilot.job.estimated_cost_hourly"].append((hourly_cost, gpu_tags))

            if job_status == "queued":
                submitted_ts = job.get("submitted_at", 0)
                if submitted_ts > 0:
                    import time

                    wait_seconds = time.time() - submitted_ts
                    if wait_seconds > 0:
                        metrics["skypilot.job.queue_wait_seconds"].append((wait_seconds, base_tags.copy()))

            submitted_ts = job.get("submitted_at", 0)
            start_ts = job.get("start_at", 0)
            if submitted_ts > 0 and start_ts > 0 and start_ts > submitted_ts:
                setup_seconds = start_ts - submitted_ts
                metrics["skypilot.job.setup_seconds"].append((setup_seconds, base_tags.copy()))

            if job_status in ("succeeded", "failed", "cancelled"):
                end_ts = job.get("end_at", 0)
                if submitted_ts > 0 and end_ts > 0 and end_ts > submitted_ts:
                    total_duration_hours = (end_ts - submitted_ts) / 3600
                    metrics["skypilot.job.total_duration_hours"].append((total_duration_hours, base_tags.copy()))

                    accelerators = job.get("accelerators", {})
                    if isinstance(accelerators, dict):
                        for gpu_type, count in accelerators.items():
                            gpu_type_normalized = gpu_type.lower().replace(":", "_")
                            resources = job.get("cluster_resources", "")
                            instance_type = "spot" if "spot" in resources.lower() else "on_demand"
                            hourly_cost = self._estimate_gpu_cost(gpu_type, count, instance_type)

                            if hourly_cost:
                                total_cost = hourly_cost * total_duration_hours
                                gpu_tags = base_tags + [f"gpu_type:{gpu_type_normalized}"]
                                metrics["skypilot.job.total_cost_usd"].append((total_cost, gpu_tags))

            if job_status == "failed":
                if start_ts > 0 and end_ts > 0 and end_ts > start_ts:
                    time_to_failure_hours = (end_ts - start_ts) / 3600
                    metrics["skypilot.job.time_to_failure_hours"].append((time_to_failure_hours, base_tags.copy()))

        except Exception as e:
            self.logger.debug(f"Failed to emit job metrics for job {job.get('job_id', 'unknown')}: {e}")

    def _estimate_gpu_cost(self, gpu_type: str, count: int, instance_type: str) -> float | None:
        """Estimate hourly cost for GPU resources.

        Args:
            gpu_type: GPU type string (e.g., "L4", "A10G", "H100")
            count: Number of GPUs
            instance_type: "spot" or "on_demand"

        Returns:
            Estimated hourly cost in USD, or None if unknown
        """
        base_costs = {
            "l4": 0.75,
            "a10g": 1.01,
            "h100": 4.10,
            "a100": 4.72,
        }

        gpu_lower = gpu_type.lower()
        for key in base_costs:
            if key in gpu_lower:
                hourly_per_gpu = base_costs[key]
                if instance_type == "spot":
                    hourly_per_gpu *= 0.35
                return hourly_per_gpu * count

        return None
