"""Weights & Biases (WandB) metrics collector for Datadog monitoring."""

from datetime import datetime, timedelta, timezone
from typing import Any

import wandb

from devops.datadog.common.base import BaseCollector


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
        """Collect training run status metrics."""
        metrics = {
            "wandb.runs.active": 0,
            "wandb.runs.completed_7d": 0,
            "wandb.runs.failed_7d": 0,
            "wandb.runs.total": 0,
        }

        try:
            # Get all runs from the project
            runs = self.api.runs(f"{self.entity}/{self.project}")

            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            for run in runs:
                metrics["wandb.runs.total"] += 1

                # Count active runs
                if run.state == "running":
                    metrics["wandb.runs.active"] += 1

                # Count completed/failed runs in last 7 days
                if run.created_at:
                    # WandB created_at is a datetime string
                    try:
                        created_at = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        # Fallback: try parsing as timestamp
                        continue

                    if created_at >= seven_days_ago:
                        if run.state == "finished":
                            metrics["wandb.runs.completed_7d"] += 1
                        elif run.state in ("failed", "crashed"):
                            metrics["wandb.runs.failed_7d"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect run status metrics: {e}", exc_info=True)
            for key in metrics:
                metrics[key] = 0

        return metrics

    def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect model performance metrics."""
        metrics = {
            "wandb.metrics.best_accuracy": None,
            "wandb.metrics.latest_loss": None,
            "wandb.metrics.avg_accuracy_7d": None,
        }

        try:
            # Get recent completed runs
            runs = self.api.runs(f"{self.entity}/{self.project}", filters={"state": "finished"})

            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            best_accuracy = None
            latest_loss = None
            accuracies = []

            for run in runs:
                # Get run summary metrics
                summary = run.summary

                # Track best accuracy across all runs
                if "accuracy" in summary:
                    acc = summary["accuracy"]
                    if best_accuracy is None or acc > best_accuracy:
                        best_accuracy = acc

                # Track latest loss
                if "loss" in summary:
                    latest_loss = summary["loss"]

                # Collect accuracies from recent runs
                if run.created_at:
                    try:
                        created_at = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                        if created_at >= seven_days_ago and "accuracy" in summary:
                            accuracies.append(summary["accuracy"])
                    except (ValueError, AttributeError):
                        continue

            metrics["wandb.metrics.best_accuracy"] = best_accuracy
            metrics["wandb.metrics.latest_loss"] = latest_loss
            if accuracies:
                metrics["wandb.metrics.avg_accuracy_7d"] = sum(accuracies) / len(accuracies)

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}", exc_info=True)

        return metrics

    def _collect_resource_metrics(self) -> dict[str, Any]:
        """Collect resource usage metrics."""
        metrics = {
            "wandb.training.avg_duration_hours": None,
            "wandb.training.gpu_utilization_avg": None,
            "wandb.training.total_gpu_hours_7d": 0.0,
        }

        try:
            runs = self.api.runs(f"{self.entity}/{self.project}")

            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            durations = []
            gpu_utils = []
            total_gpu_hours = 0.0

            for run in runs:
                # Calculate duration
                if run.created_at and run.heartbeat_at:
                    try:
                        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                        heartbeat = datetime.fromisoformat(run.heartbeat_at.replace("Z", "+00:00"))
                        duration_hours = (heartbeat - created).total_seconds() / 3600
                        durations.append(duration_hours)

                        # Track GPU hours for recent runs
                        if created >= seven_days_ago:
                            # Estimate GPU hours (assuming 1 GPU per run)
                            # This is a simplification - real implementation would check run config
                            total_gpu_hours += duration_hours

                    except (ValueError, AttributeError):
                        continue

                # Get GPU utilization from summary metrics
                summary = run.summary
                if "system.gpu.0.gpu" in summary:
                    gpu_utils.append(summary["system.gpu.0.gpu"])
                elif "gpu_util" in summary:
                    gpu_utils.append(summary["gpu_util"])

            if durations:
                metrics["wandb.training.avg_duration_hours"] = sum(durations) / len(durations)

            if gpu_utils:
                metrics["wandb.training.gpu_utilization_avg"] = sum(gpu_utils) / len(gpu_utils)

            metrics["wandb.training.total_gpu_hours_7d"] = total_gpu_hours

        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}", exc_info=True)

        return metrics
