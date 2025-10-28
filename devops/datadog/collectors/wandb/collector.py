"""Weights & Biases (WandB) metrics collector for Datadog monitoring."""

import json
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
                # Sweep runs: either WandB sweep attribute or ".sweep" in name
                elif run.sweep or ".sweep" in run.name.lower():
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
            metrics.update(self._collect_sweep_metrics(sweep_runs))

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
            "wandb.push_to_main.overview.steps_per_second": None,
            "wandb.push_to_main.overview.epoch_steps_per_second": None,
            "wandb.push_to_main.overview.sps": None,
            "wandb.push_to_main.timing_cumulative.sps": None,
            "wandb.push_to_main.timing_per_epoch.sps": None,
            "wandb.push_to_main.avg_duration_hours": None,
            "wandb.push_to_main.heart.gained": None,
            "wandb.push_to_main.skypilot.queue_latency_s": None,
        }

        if not runs:
            self.logger.info("No github.sky push-to-main runs found in last 24h")
            return metrics

        try:
            completed = 0
            failed = 0
            # Track each SPS metric separately
            overview_sps_values = []
            overview_epoch_sps_values = []
            overview_sps_alt_values = []
            timing_cumulative_sps_values = []
            timing_per_epoch_sps_values = []
            durations = []
            # Track heart and latency metrics
            heart_gained_values = []
            queue_latency_values = []

            for run in runs:
                # Count by state
                if run.state == "finished":
                    completed += 1
                elif run.state in ["failed", "crashed"]:
                    failed += 1

                # Extract metrics from finished OR crashed runs
                # Crashed runs still have valuable summary metrics
                if run.state in ["finished", "crashed"]:
                    try:
                        # Handle WandB API quirk: crashed runs have _json_dict as JSON string
                        if hasattr(run.summary, "_json_dict"):
                            json_dict_value = run.summary._json_dict
                            if isinstance(json_dict_value, str):
                                # Parse JSON string for crashed runs
                                summary_dict = json.loads(json_dict_value)
                            else:
                                # Use dict directly for finished runs
                                summary_dict = json_dict_value
                        else:
                            # Fallback to dict() conversion
                            summary_dict = dict(run.summary)

                        # Collect all SPS metrics
                        if "overview/steps_per_second" in summary_dict:
                            overview_sps_values.append(summary_dict["overview/steps_per_second"])
                        if "overview/epoch_steps_per_second" in summary_dict:
                            overview_epoch_sps_values.append(summary_dict["overview/epoch_steps_per_second"])
                        if "overview/sps" in summary_dict:
                            overview_sps_alt_values.append(summary_dict["overview/sps"])
                        if "timing_cumulative/sps" in summary_dict:
                            timing_cumulative_sps_values.append(summary_dict["timing_cumulative/sps"])
                        if "timing_per_epoch/sps" in summary_dict:
                            timing_per_epoch_sps_values.append(summary_dict["timing_per_epoch/sps"])

                        # Collect heart and latency metrics
                        if "env_agent/heart.gained" in summary_dict:
                            heart_gained_values.append(summary_dict["env_agent/heart.gained"])
                        if "skypilot/queue_latency_s" in summary_dict:
                            queue_latency_values.append(summary_dict["skypilot/queue_latency_s"])

                    except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
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

            # Average each SPS metric (use latest value from most recent run)
            if overview_sps_values:
                metrics["wandb.push_to_main.overview.steps_per_second"] = overview_sps_values[-1]
            if overview_epoch_sps_values:
                metrics["wandb.push_to_main.overview.epoch_steps_per_second"] = overview_epoch_sps_values[-1]
            if overview_sps_alt_values:
                metrics["wandb.push_to_main.overview.sps"] = overview_sps_alt_values[-1]
            if timing_cumulative_sps_values:
                metrics["wandb.push_to_main.timing_cumulative.sps"] = timing_cumulative_sps_values[-1]
            if timing_per_epoch_sps_values:
                metrics["wandb.push_to_main.timing_per_epoch.sps"] = timing_per_epoch_sps_values[-1]

            if durations:
                metrics["wandb.push_to_main.avg_duration_hours"] = sum(durations) / len(durations)

            # Heart and latency metrics (use latest value from most recent run)
            if heart_gained_values:
                metrics["wandb.push_to_main.heart.gained"] = heart_gained_values[-1]
            if queue_latency_values:
                metrics["wandb.push_to_main.skypilot.queue_latency_s"] = queue_latency_values[-1]

            self.logger.info(f"Push-to-main: {len(runs)} runs ({completed} completed, {failed} failed)")

        except Exception as e:
            self.logger.error(f"Failed to collect push-to-main metrics: {e}", exc_info=True)

        return metrics

    def _collect_sweep_metrics(self, runs: list) -> dict[str, Any]:
        """Collect metrics specifically for sweep/hyperparameter tuning runs.

        Sweep runs have ".sweep" in the name or WandB sweep attribute set.

        Args:
            runs: List of sweep runs (filtered by name pattern or sweep attribute)

        Returns:
            Dictionary of sweep-specific metrics
        """
        metrics = {
            "wandb.sweep.runs_total_24h": 0,
            "wandb.sweep.runs_completed_24h": 0,
            "wandb.sweep.runs_failed_24h": 0,
            "wandb.sweep.runs_active": 0,
            "wandb.sweep.success_rate_pct": None,
        }

        if not runs:
            self.logger.info("No sweep runs found in last 24h")
            return metrics

        try:
            completed = 0
            failed = 0
            active = 0

            for run in runs:
                # Count by state
                if run.state == "finished":
                    completed += 1
                elif run.state in ["failed", "crashed"]:
                    failed += 1
                elif run.state == "running":
                    active += 1

            # Calculate metrics
            metrics["wandb.sweep.runs_total_24h"] = len(runs)
            metrics["wandb.sweep.runs_completed_24h"] = completed
            metrics["wandb.sweep.runs_failed_24h"] = failed
            metrics["wandb.sweep.runs_active"] = active

            total_finished = completed + failed
            if total_finished > 0:
                metrics["wandb.sweep.success_rate_pct"] = (completed / total_finished) * 100

            self.logger.info(f"Sweep: {len(runs)} runs ({completed} completed, {failed} failed, {active} active)")

        except Exception as e:
            self.logger.error(f"Failed to collect sweep metrics: {e}", exc_info=True)

        return metrics
