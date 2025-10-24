"""Weights & Biases (WandB) metrics collector for Datadog monitoring."""

from datetime import datetime, timedelta, timezone
from typing import Any

import wandb

from devops.datadog.utils.base import BaseCollector


class WandBCollector(BaseCollector):
    """Collector for WandB training run metrics.

    Collects metrics about training runs, model performance, and resource usage
    from Weights & Biases. Fetches recent runs once and categorizes them for
    efficient metric collection.
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

        Fetches recent runs once and categorizes them for efficient processing.

        Returns:
            Dictionary mapping metric keys to values
        """
        metrics = {}

        try:
            # Fetch all recent runs once (last 24 hours)
            one_day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_runs = list(
                self.api.runs(
                    f"{self.entity}/{self.project}",
                    filters={"created_at": {"$gte": one_day_ago.isoformat()}},
                )
            )

            self.logger.info(f"Fetched {len(recent_runs)} runs from last 24h")

            # Categorize runs by type
            push_to_main_runs = []
            sweep_runs = []
            regular_runs = []

            for run in recent_runs:
                if not run.name:
                    continue

                # GitHub CI push-to-main runs: github.sky.pr* or github.sky.main.*
                if run.name.startswith("github.sky."):
                    push_to_main_runs.append(run)
                # Sweep runs (if you have a naming pattern for sweeps)
                elif run.sweep:
                    sweep_runs.append(run)
                # Everything else
                else:
                    regular_runs.append(run)

            self.logger.info(
                f"Categorized: {len(push_to_main_runs)} push-to-main, "
                f"{len(sweep_runs)} sweep, {len(regular_runs)} regular"
            )

            # Collect metrics from categorized runs
            metrics.update(self._collect_overall_metrics(recent_runs))
            metrics.update(self._collect_push_to_main_metrics(push_to_main_runs))
            # Could add: metrics.update(self._collect_sweep_metrics(sweep_runs))

        except Exception as e:
            self.logger.error(f"Failed to collect WandB metrics: {e}", exc_info=True)

        return metrics

    def _collect_overall_metrics(self, runs: list) -> dict[str, Any]:
        """Collect overall metrics from all recent runs.

        Args:
            runs: List of all recent WandB runs

        Returns:
            Dictionary of overall metrics
        """
        metrics = {
            "wandb.runs.active": 0,
            "wandb.runs.completed_24h": 0,
            "wandb.runs.failed_24h": 0,
            "wandb.runs.total_recent": len(runs),
            "wandb.training.avg_duration_hours": None,
            "wandb.training.total_gpu_hours_24h": 0.0,
        }

        try:
            durations = []
            total_gpu_hours = 0.0

            for run in runs:
                # Count by state
                if run.state == "running":
                    metrics["wandb.runs.active"] += 1
                elif run.state == "finished":
                    metrics["wandb.runs.completed_24h"] += 1
                elif run.state in ["failed", "crashed"]:
                    metrics["wandb.runs.failed_24h"] += 1

                # Calculate duration
                if run.created_at and run.heartbeat_at:
                    try:
                        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                        heartbeat = datetime.fromisoformat(run.heartbeat_at.replace("Z", "+00:00"))
                        duration_hours = (heartbeat - created).total_seconds() / 3600
                        durations.append(duration_hours)
                        # Estimate GPU hours (assuming 1 GPU per run - could be enhanced)
                        total_gpu_hours += duration_hours
                    except (ValueError, AttributeError):
                        continue

            if durations:
                metrics["wandb.training.avg_duration_hours"] = sum(durations) / len(durations)

            metrics["wandb.training.total_gpu_hours_24h"] = total_gpu_hours

        except Exception as e:
            self.logger.error(f"Failed to collect overall metrics: {e}", exc_info=True)

        return metrics

    def _collect_push_to_main_metrics(self, runs: list) -> dict[str, Any]:
        """Collect metrics specifically for GitHub CI push-to-main runs.

        These runs follow the naming pattern: github.sky.pr{NUMBER}.{COMMIT}.{ENV}.{TIMESTAMP}
        They are the most important to track as they represent the baseline performance
        of the main branch over time.

        Args:
            runs: List of push-to-main runs (filtered by name pattern)

        Returns:
            Dictionary of push-to-main specific metrics
        """
        metrics = {
            "wandb.push_to_main.runs_completed_24h": 0,
            "wandb.push_to_main.runs_failed_24h": 0,
            "wandb.push_to_main.success_rate_pct": None,
            "wandb.push_to_main.avg_steps_per_second": None,
            "wandb.push_to_main.latest_steps_per_second": None,
            "wandb.push_to_main.avg_duration_hours": None,
        }

        if not runs:
            self.logger.info("No github.sky push-to-main runs found in last 24h")
            return metrics

        try:
            completed = 0
            failed = 0
            sps_values = []
            durations = []

            for run in runs:
                # Count by state
                if run.state == "finished":
                    completed += 1
                elif run.state in ["failed", "crashed"]:
                    failed += 1

                # Extract metrics from completed runs
                if run.state == "finished":
                    try:
                        summary_dict = dict(run.summary)

                        # SPS (steps per second) - training throughput
                        if "overview/steps_per_second" in summary_dict:
                            sps_values.append(summary_dict["overview/steps_per_second"])
                        elif "overview/sps" in summary_dict:
                            sps_values.append(summary_dict["overview/sps"])

                    except (TypeError, ValueError, AttributeError):
                        continue

                # Calculate duration
                if run.created_at and run.heartbeat_at:
                    try:
                        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                        heartbeat = datetime.fromisoformat(run.heartbeat_at.replace("Z", "+00:00"))
                        duration_hours = (heartbeat - created).total_seconds() / 3600
                        durations.append(duration_hours)
                    except (ValueError, AttributeError):
                        continue

            # Calculate metrics
            metrics["wandb.push_to_main.runs_completed_24h"] = completed
            metrics["wandb.push_to_main.runs_failed_24h"] = failed

            total_runs = completed + failed
            if total_runs > 0:
                metrics["wandb.push_to_main.success_rate_pct"] = (completed / total_runs) * 100

            if sps_values:
                metrics["wandb.push_to_main.avg_steps_per_second"] = sum(sps_values) / len(sps_values)
                metrics["wandb.push_to_main.latest_steps_per_second"] = sps_values[-1]  # Most recent

            if durations:
                metrics["wandb.push_to_main.avg_duration_hours"] = sum(durations) / len(durations)

            self.logger.info(f"Push-to-main: {len(runs)} runs ({completed} completed, {failed} failed)")

        except Exception as e:
            self.logger.error(f"Failed to collect push-to-main metrics: {e}", exc_info=True)

        return metrics
