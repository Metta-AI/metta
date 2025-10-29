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
        """Collect all Skypilot metrics."""
        metrics = {}
        metrics.update(self._collect_cluster_metrics())
        metrics.update(self._collect_job_metrics())
        return metrics

    def _collect_cluster_metrics(self) -> dict[str, Any]:
        """Collect cluster-related metrics."""
        metrics = {}

        try:
            # Get cluster status
            request_id = sky.status(all_users=True)
            clusters = sky.get(request_id)

            # Count active clusters
            active_clusters = sum(1 for c in clusters if c.get("status") == sky.ClusterStatus.UP)
            metrics["skypilot.clusters.active"] = active_clusters

        except Exception as e:
            self.logger.error(f"Failed to collect cluster metrics: {e}")
            metrics["skypilot.clusters.active"] = None

        return metrics

    def _collect_job_metrics(self) -> dict[str, Any]:
        """Collect comprehensive job-related metrics."""
        metrics = {
            # Job status counts
            "skypilot.jobs.queued": 0,
            "skypilot.jobs.running": 0,
            "skypilot.jobs.failed": 0,
            "skypilot.jobs.succeeded": 0,
            "skypilot.jobs.cancelled": 0,
            "skypilot.jobs.failed_7d": 0,
            # Runtime statistics (for running jobs)
            "skypilot.jobs.runtime_seconds.min": None,
            "skypilot.jobs.runtime_seconds.max": None,
            "skypilot.jobs.runtime_seconds.avg": None,
            "skypilot.jobs.runtime_seconds.p50": None,
            "skypilot.jobs.runtime_seconds.p90": None,
            "skypilot.jobs.runtime_seconds.p99": None,
            # Runtime buckets (for histogram)
            "skypilot.jobs.runtime_buckets.0_1h": 0,
            "skypilot.jobs.runtime_buckets.1_4h": 0,
            "skypilot.jobs.runtime_buckets.4_24h": 0,
            "skypilot.jobs.runtime_buckets.over_24h": 0,
            # Resource utilization
            "skypilot.resources.gpus.l4_count": 0,
            "skypilot.resources.gpus.a10g_count": 0,
            "skypilot.resources.gpus.h100_count": 0,
            "skypilot.resources.gpus.total_count": 0,
            "skypilot.resources.spot_jobs": 0,
            "skypilot.resources.ondemand_jobs": 0,
            # Reliability metrics
            "skypilot.jobs.with_recoveries": 0,
            "skypilot.jobs.recovery_count.avg": None,
            "skypilot.jobs.recovery_count.max": None,
            # Regional distribution
            "skypilot.regions.us_east_1": 0,
            "skypilot.regions.us_west_2": 0,
            "skypilot.regions.other": 0,
            # User activity
            "skypilot.users.active_count": 0,
            # Per-job metrics with tags (list of tuples)
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
            # Get job queue
            request_id = sky.jobs.queue(refresh=False, all_users=True)
            jobs = sky.get(request_id)

            now = datetime.datetime.now(datetime.timezone.utc)
            seven_days_ago = now - datetime.timedelta(days=7)

            # Collect data for statistics
            running_durations = []
            recovery_counts = []
            active_users = set()

            for job in jobs:
                status = job.get("status")
                user = job.get("user_name")
                if user:
                    active_users.add(user)

                # Count by current status
                if status == sky.jobs.ManagedJobStatus.PENDING:
                    metrics["skypilot.jobs.queued"] += 1
                    # Emit per-job metrics for queued jobs
                    self._emit_job_metrics(job, "queued", metrics)

                elif status == sky.jobs.ManagedJobStatus.RUNNING:
                    metrics["skypilot.jobs.running"] += 1

                    # Runtime statistics for running jobs
                    duration = job.get("job_duration", 0)
                    if duration > 0:
                        running_durations.append(duration)

                        # Runtime buckets
                        hours = duration / 3600
                        if hours < 1:
                            metrics["skypilot.jobs.runtime_buckets.0_1h"] += 1
                        elif hours < 4:
                            metrics["skypilot.jobs.runtime_buckets.1_4h"] += 1
                        elif hours < 24:
                            metrics["skypilot.jobs.runtime_buckets.4_24h"] += 1
                        else:
                            metrics["skypilot.jobs.runtime_buckets.over_24h"] += 1

                    # Resource tracking for running jobs
                    accelerators = job.get("accelerators", {})
                    if isinstance(accelerators, dict):
                        for gpu_type, count in accelerators.items():
                            gpu_type_lower = gpu_type.lower()
                            if "l4" in gpu_type_lower:
                                metrics["skypilot.resources.gpus.l4_count"] += count
                            elif "a10g" in gpu_type_lower:
                                metrics["skypilot.resources.gpus.a10g_count"] += count
                            elif "h100" in gpu_type_lower:
                                metrics["skypilot.resources.gpus.h100_count"] += count
                            metrics["skypilot.resources.gpus.total_count"] += count

                    # Spot vs on-demand
                    resources_str = job.get("cluster_resources", "")
                    if "spot" in resources_str.lower():
                        metrics["skypilot.resources.spot_jobs"] += 1
                    else:
                        metrics["skypilot.resources.ondemand_jobs"] += 1

                    # Regional distribution
                    region = job.get("region", "")
                    if "us-east-1" in region:
                        metrics["skypilot.regions.us_east_1"] += 1
                    elif "us-west-2" in region:
                        metrics["skypilot.regions.us_west_2"] += 1
                    elif region:
                        metrics["skypilot.regions.other"] += 1

                    # Recovery tracking
                    recovery_count = job.get("recovery_count", 0)
                    if recovery_count > 0:
                        metrics["skypilot.jobs.with_recoveries"] += 1
                        recovery_counts.append(recovery_count)

                    # Emit per-job metrics with tags
                    self._emit_job_metrics(job, "running", metrics)

                elif status in (sky.jobs.ManagedJobStatus.FAILED, sky.jobs.ManagedJobStatus.FAILED_SETUP):
                    metrics["skypilot.jobs.failed"] += 1

                    # Check if failed in last 7 days
                    submitted_ts = job.get("submitted_at", 0)
                    submitted_at = datetime.datetime.fromtimestamp(submitted_ts, tz=datetime.timezone.utc)
                    if submitted_at >= seven_days_ago:
                        metrics["skypilot.jobs.failed_7d"] += 1

                    # Emit metrics for failed jobs
                    self._emit_job_metrics(job, "failed", metrics)

                elif status == sky.jobs.ManagedJobStatus.SUCCEEDED:
                    metrics["skypilot.jobs.succeeded"] += 1
                    # Emit metrics for completed jobs
                    self._emit_job_metrics(job, "succeeded", metrics)

                elif status == sky.jobs.ManagedJobStatus.CANCELLED:
                    metrics["skypilot.jobs.cancelled"] += 1
                    # Emit metrics for cancelled jobs
                    self._emit_job_metrics(job, "cancelled", metrics)

            # Calculate runtime statistics
            if running_durations:
                running_durations.sort()
                n = len(running_durations)

                metrics["skypilot.jobs.runtime_seconds.min"] = running_durations[0]
                metrics["skypilot.jobs.runtime_seconds.max"] = running_durations[-1]
                metrics["skypilot.jobs.runtime_seconds.avg"] = sum(running_durations) / n
                metrics["skypilot.jobs.runtime_seconds.p50"] = running_durations[int(n * 0.50)]
                metrics["skypilot.jobs.runtime_seconds.p90"] = running_durations[int(n * 0.90)]
                metrics["skypilot.jobs.runtime_seconds.p99"] = running_durations[min(int(n * 0.99), n - 1)]

            # Calculate recovery statistics
            if recovery_counts:
                metrics["skypilot.jobs.recovery_count.avg"] = sum(recovery_counts) / len(recovery_counts)
                metrics["skypilot.jobs.recovery_count.max"] = max(recovery_counts)

            # Active user count
            metrics["skypilot.users.active_count"] = len(active_users)

        except sky.exceptions.ClusterNotUpError:
            self.logger.warning("Jobs controller is not up, skipping job metrics")
            # Set all metrics to None when jobs controller is down
            for key in metrics:
                metrics[key] = None
        except Exception as e:
            self.logger.error(f"Failed to collect job metrics: {e}")
            # Set all metrics to None on error
            for key in metrics:
                metrics[key] = None

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

            # Determine instance type (spot vs on-demand)
            resources_str = job.get("cluster_resources", "")
            instance_type = "spot" if "spot" in resources_str.lower() else "on_demand"

            # Base tags for this job
            base_tags = [
                f"job_id:{job_id}",
                f"user:{user}",
                f"region:{region}",
                f"instance_type:{instance_type}",
                f"status:{job_status}",
            ]

            # Runtime hours (for both queued and running)
            duration_seconds = job.get("job_duration", 0)
            if duration_seconds > 0:
                runtime_hours = duration_seconds / 3600
                metrics["skypilot.job.runtime_hours"].append((runtime_hours, base_tags.copy()))

            # GPU metrics
            accelerators = job.get("accelerators", {})
            if isinstance(accelerators, dict):
                for gpu_type, count in accelerators.items():
                    gpu_type_normalized = gpu_type.lower().replace(":", "_")
                    gpu_tags = base_tags + [f"gpu_type:{gpu_type_normalized}"]

                    # GPU count per job
                    metrics["skypilot.job.gpu_count"].append((count, gpu_tags))

                    # Estimated hourly cost (rough estimates)
                    hourly_cost = self._estimate_gpu_cost(gpu_type, count, instance_type)
                    if hourly_cost:
                        metrics["skypilot.job.estimated_cost_hourly"].append((hourly_cost, gpu_tags))

            # Queue wait time (for queued jobs)
            if job_status == "queued":
                submitted_ts = job.get("submitted_at", 0)
                if submitted_ts > 0:
                    import time

                    wait_seconds = time.time() - submitted_ts
                    if wait_seconds > 0:
                        metrics["skypilot.job.queue_wait_seconds"].append((wait_seconds, base_tags.copy()))

            # Setup latency (time from submit to start) - for running/completed/failed jobs
            submitted_ts = job.get("submitted_at", 0)
            start_ts = job.get("start_at", 0)
            if submitted_ts > 0 and start_ts > 0 and start_ts > submitted_ts:
                setup_seconds = start_ts - submitted_ts
                metrics["skypilot.job.setup_seconds"].append((setup_seconds, base_tags.copy()))

            # Total duration and cost (for completed/failed/cancelled jobs)
            if job_status in ("succeeded", "failed", "cancelled"):
                end_ts = job.get("end_at", 0)
                if submitted_ts > 0 and end_ts > 0 and end_ts > submitted_ts:
                    total_duration_hours = (end_ts - submitted_ts) / 3600
                    metrics["skypilot.job.total_duration_hours"].append((total_duration_hours, base_tags.copy()))

                    # Calculate total cost (duration * hourly rate)
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

            # Time to failure (for failed jobs specifically)
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
        # Rough cost estimates (USD/hour per GPU, AWS prices as of 2025)
        # These are approximations - actual costs vary by region and availability
        base_costs = {
            "l4": 0.75,  # g6 instances
            "a10g": 1.01,  # g5 instances
            "h100": 4.10,  # p5 instances
            "a100": 4.72,  # p4d instances
        }

        # Normalize GPU type
        gpu_lower = gpu_type.lower()
        for key in base_costs:
            if key in gpu_lower:
                hourly_per_gpu = base_costs[key]
                # Spot instances are typically 50-70% cheaper
                if instance_type == "spot":
                    hourly_per_gpu *= 0.35
                return hourly_per_gpu * count

        return None
