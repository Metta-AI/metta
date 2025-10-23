"""Skypilot metrics collector for Datadog monitoring."""

import datetime
from typing import Any

import sky
import sky.jobs

from devops.datadog.common.base import BaseCollector


class SkypilotCollector(BaseCollector):
    """Collector for Skypilot cluster and job metrics.

    Collects comprehensive metrics about active clusters, job statuses, runtime
    distributions, resource utilization, and reliability from the Skypilot job
    orchestration system.
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

                elif status in (sky.jobs.ManagedJobStatus.FAILED, sky.jobs.ManagedJobStatus.FAILED_SETUP):
                    metrics["skypilot.jobs.failed"] += 1

                    # Check if failed in last 7 days
                    submitted_ts = job.get("submitted_at", 0)
                    submitted_at = datetime.datetime.fromtimestamp(submitted_ts, tz=datetime.timezone.utc)
                    if submitted_at >= seven_days_ago:
                        metrics["skypilot.jobs.failed_7d"] += 1

                elif status == sky.jobs.ManagedJobStatus.SUCCEEDED:
                    metrics["skypilot.jobs.succeeded"] += 1

                elif status == sky.jobs.ManagedJobStatus.CANCELLED:
                    metrics["skypilot.jobs.cancelled"] += 1

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
