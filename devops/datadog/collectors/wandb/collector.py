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

        Fetches recent runs and emits per-run metrics with tags for Datadog aggregation.
        Datadog deduplicates by run_id tag automatically.

        Returns:
            Dictionary mapping metric keys to lists of (value, tags) tuples
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

            # Process each run individually - emit per-run metrics
            for run in recent_runs:
                if not run.name:
                    continue

                # Determine run type (order matters - check specific patterns first)
                if run.name.startswith("github.sky."):
                    run_type = "ptm"  # GitHub CI push-to-main
                elif "sweep" in run.name.lower() or run.sweep:
                    run_type = "sweep"  # Hyperparameter sweep
                elif "stable" in run.name.lower():
                    run_type = "stable"  # Stable baseline runs
                elif "local" in run.name.lower():
                    run_type = "local"  # Local development runs
                else:
                    run_type = "other"  # Catch-all

                # Determine state
                if run.state == "running":
                    state = "active"
                elif run.state == "finished":
                    state = "success"
                elif run.state in ["failed", "crashed"]:
                    state = "failure"
                else:
                    continue  # Skip unknown states

                # Base tags for this run
                tags = [f"run_id:{run.id}", f"run_type:{run_type}", f"state:{state}"]

                # Emit metrics for all runs (including active)
                self._emit_run_metrics(run, run_type, state, tags, metrics)

            active_count = sum(1 for r in recent_runs if r.state == "running")
            self.logger.info(f"Processed {len(recent_runs)} runs, {active_count} active")

        except Exception as e:
            self.logger.error(f"Failed to collect WandB metrics: {e}", exc_info=True)

        return metrics

    def _emit_run_metrics(self, run: Any, run_type: str, state: str, tags: list[str], metrics: dict[str, Any]) -> None:
        """Extract and emit metrics for a single run.

        Emits per-run instantaneous values. Datadog will aggregate these.

        Args:
            run: WandB run object
            run_type: Run type (ptm, sweep, stable, local, other)
            state: Run state (success, failure, active)
            tags: Base tags for this run
            metrics: Dictionary to append metrics to
        """
        try:
            # Calculate duration (elapsed time for active runs, total for completed)
            duration_hours = None
            if run.created_at and run.heartbeat_at:
                try:
                    created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
                    heartbeat = datetime.fromisoformat(run.heartbeat_at.replace("Z", "+00:00"))
                    duration_hours = (heartbeat - created).total_seconds() / 3600
                except (ValueError, AttributeError):
                    pass

            # Emit duration if available (all states including active)
            if duration_hours is not None:
                metric_key = f"wandb.{run_type}.{state}.duration_hours"
                if metric_key not in metrics:
                    metrics[metric_key] = []
                metrics[metric_key].append((duration_hours, tags))

            # For active runs, we only have duration - no final metrics yet
            if state == "active":
                return

            # Extract summary metrics (for finished/crashed runs only)
            try:
                # Handle WandB API quirk: crashed runs have _json_dict as JSON string
                if hasattr(run.summary, "_json_dict"):
                    json_dict_value = run.summary._json_dict
                    if isinstance(json_dict_value, str):
                        summary_dict = json.loads(json_dict_value)
                    else:
                        summary_dict = json_dict_value
                else:
                    summary_dict = dict(run.summary)

                # SPS metrics (prefer overview/steps_per_second)
                sps = None
                if "overview/steps_per_second" in summary_dict:
                    sps = summary_dict["overview/steps_per_second"]
                elif "overview/sps" in summary_dict:
                    sps = summary_dict["overview/sps"]

                if sps is not None:
                    metric_key = f"wandb.{run_type}.{state}.sps"
                    if metric_key not in metrics:
                        metrics[metric_key] = []
                    metrics[metric_key].append((sps, tags))

                # Hearts gained (PTM runs)
                if run_type == "ptm" and "env_agent/heart.gained" in summary_dict:
                    hearts = summary_dict["env_agent/heart.gained"]
                    metric_key = f"wandb.{run_type}.{state}.hearts_gained"
                    if metric_key not in metrics:
                        metrics[metric_key] = []
                    metrics[metric_key].append((hearts, tags))

                # SkyPilot queue latency (PTM runs)
                if run_type == "ptm" and "skypilot/queue_latency_s" in summary_dict:
                    latency = summary_dict["skypilot/queue_latency_s"]
                    metric_key = f"wandb.{run_type}.{state}.skypilot_latency_s"
                    if metric_key not in metrics:
                        metrics[metric_key] = []
                    metrics[metric_key].append((latency, tags))

            except (TypeError, ValueError, AttributeError, json.JSONDecodeError) as e:
                self.logger.debug(f"Could not extract summary metrics from run {run.id}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to emit metrics for run {run.id}: {e}", exc_info=True)
