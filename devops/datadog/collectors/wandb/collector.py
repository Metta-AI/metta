"""Weights & Biases (WandB) metrics collector for Datadog monitoring."""

from datetime import datetime, timedelta, timezone
from typing import Any

import wandb

from devops.datadog.utils.base import BaseCollector


class WandBCollector(BaseCollector):
    """Collector for WandB training run metrics.

    Collects metrics about training runs, model performance, and resource usage
    from Weights & Biases.
    """

    def __init__(self, api_key: str, entity: str, project: str):
        """Initialize WandB collector.

        Args:
            api_key: WandB API key for authentication
            entity: WandB entity (username or team name)
            project: WandB project name to monitor
        """
        super().__init__(name="wandb")
        self.entity = entity
        self.project = project

        # Initialize WandB API client
        wandb.login(key=api_key)
        self.api = wandb.Api()

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all WandB metrics.

        Returns:
            Dictionary mapping metric keys to values
        """
        metrics = {}

        # Collect run status metrics
        metrics.update(self._collect_run_status_metrics())

        # Collect performance metrics
        metrics.update(self._collect_performance_metrics())

        # Collect resource usage metrics
        metrics.update(self._collect_resource_metrics())

        return metrics

    def _collect_run_status_metrics(self) -> dict[str, Any]:
        """Collect training run status metrics.

        Focus on recent runs (last 24 hours) to avoid fetching 26k+ historical runs.
        This provides actionable monitoring data without performance issues.
        """
        metrics = {
            "wandb.runs.active": 0,
            "wandb.runs.completed_24h": 0,
            "wandb.runs.failed_24h": 0,
            "wandb.runs.total_recent": 0,
        }

        try:
            # Count active runs using filter
            active_runs = self.api.runs(f"{self.entity}/{self.project}", filters={"state": "running"})
            metrics["wandb.runs.active"] = len(list(active_runs))

            # Focus on last 24 hours for actionable monitoring
            one_day_ago = datetime.now(timezone.utc) - timedelta(hours=24)

            # Count completed runs in last 24 hours using filter
            recent_completed = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={"state": "finished", "created_at": {"$gte": one_day_ago.isoformat()}},
            )
            metrics["wandb.runs.completed_24h"] = len(list(recent_completed))

            # Count failed runs in last 24 hours using filter
            recent_failed = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={
                    "$or": [{"state": "failed"}, {"state": "crashed"}],
                    "created_at": {"$gte": one_day_ago.isoformat()},
                },
            )
            metrics["wandb.runs.failed_24h"] = len(list(recent_failed))

            # Total recent activity (last 24h + currently active)
            metrics["wandb.runs.total_recent"] = (
                metrics["wandb.runs.active"] + metrics["wandb.runs.completed_24h"] + metrics["wandb.runs.failed_24h"]
            )

        except Exception as e:
            self.logger.error(f"Failed to collect run status metrics: {e}", exc_info=True)
            for key in metrics:
                metrics[key] = 0

        return metrics

    def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect model performance metrics from recent runs.

        Focuses on GitHub CI runs (pattern: github.sky.main.*) to track training performance
        from automated CI runs on main branch.
        """
        metrics = {
            "wandb.metrics.latest_sps": None,  # Steps per second (training throughput)
            "wandb.metrics.avg_heart_amount_24h": None,  # Average heart amount (survival metric)
            "wandb.metrics.latest_queue_latency_s": None,  # SkyPilot queue latency
        }

        try:
            # Only fetch recent completed runs (last 24 hours)
            # Focus on GitHub CI runs: github.sky.main.*
            one_day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={"state": "finished", "created_at": {"$gte": one_day_ago.isoformat()}},
            )

            latest_sps = None
            heart_amounts = []
            latest_queue_latency = None

            for run in recent_runs:
                # Get run summary metrics
                # WandB summary objects can be complex and may not convert to dict properly
                try:
                    summary_dict = dict(run.summary)
                except (TypeError, ValueError, AttributeError):
                    # If summary cannot be converted to dict, skip this run
                    continue

                # Track latest SPS (steps per second) - primary training throughput metric
                try:
                    if "overview/sps" in summary_dict:
                        latest_sps = summary_dict["overview/sps"]
                except (TypeError, KeyError):
                    pass

                # Track heart amount (agent survival metric)
                try:
                    if "env_agent/heart.amount" in summary_dict:
                        heart_amounts.append(summary_dict["env_agent/heart.amount"])
                except (TypeError, KeyError):
                    pass

                # Track SkyPilot queue latency
                try:
                    if "skypilot/queue_latency_s" in summary_dict:
                        latest_queue_latency = summary_dict["skypilot/queue_latency_s"]
                except (TypeError, KeyError):
                    pass

            metrics["wandb.metrics.latest_sps"] = latest_sps
            if heart_amounts:
                metrics["wandb.metrics.avg_heart_amount_24h"] = sum(heart_amounts) / len(heart_amounts)
            metrics["wandb.metrics.latest_queue_latency_s"] = latest_queue_latency

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}", exc_info=True)

        return metrics

    def _collect_resource_metrics(self) -> dict[str, Any]:
        """Collect resource usage metrics from recent runs."""
        metrics = {
            "wandb.training.avg_duration_hours": None,
            "wandb.training.gpu_utilization_avg": None,
            "wandb.training.total_gpu_hours_24h": 0.0,
        }

        try:
            # Only fetch recent runs (last 24 hours) for resource metrics
            one_day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters={"created_at": {"$gte": one_day_ago.isoformat()}},
            )

            durations = []
            gpu_utils = []
            total_gpu_hours = 0.0

            for run in recent_runs:
                # Calculate duration
                if run.created_at and run.heartbeat_at:
                    try:
                        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                        heartbeat = datetime.fromisoformat(run.heartbeat_at.replace("Z", "+00:00"))
                        duration_hours = (heartbeat - created).total_seconds() / 3600
                        durations.append(duration_hours)

                        # Estimate GPU hours (assuming 1 GPU per run)
                        # This is a simplification - real implementation would check run config
                        total_gpu_hours += duration_hours

                    except (ValueError, AttributeError):
                        continue

                # Get GPU utilization from summary metrics
                # Use dict() to safely access summary data
                try:
                    summary_dict = dict(run.summary)

                    if "system.gpu.0.gpu" in summary_dict:
                        gpu_utils.append(summary_dict["system.gpu.0.gpu"])
                    elif "gpu_util" in summary_dict:
                        gpu_utils.append(summary_dict["gpu_util"])
                except (TypeError, ValueError, AttributeError, KeyError):
                    # Skip runs with invalid summary data
                    continue

            if durations:
                metrics["wandb.training.avg_duration_hours"] = sum(durations) / len(durations)

            if gpu_utils:
                metrics["wandb.training.gpu_utilization_avg"] = sum(gpu_utils) / len(gpu_utils)

            metrics["wandb.training.total_gpu_hours_24h"] = total_gpu_hours

        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}", exc_info=True)

        return metrics
