"""
Statistics collection and logging for Mock Dynamical System Environment.

This module provides enhanced stats logging specifically for the mock environment
to track task dependency dynamics and learning progress in wandb.
"""

from typing import Any, Dict

from metta.rl.training import TrainerComponent


class MockEnvironmentStatsLogger(TrainerComponent):
    """Collects and logs mock environment specific statistics to wandb."""

    def __init__(self, wandb_run, epoch_interval: int = 1):
        super().__init__(epoch_interval=epoch_interval)
        self._wandb_run = wandb_run
        self._last_stats = {}

    def process_rollout(self, raw_infos: list[dict[str, Any]]) -> None:
        """Process rollout information to extract mock environment stats."""
        # Aggregate stats from all environment info dicts
        if not raw_infos:
            return

        # Get the latest info dict (should contain current state)
        latest_info = raw_infos[-1] if raw_infos else {}

        # Extract mock environment specific metrics
        mock_stats = {}

        for key, value in latest_info.items():
            if key.startswith(("task_", "mean_", "max_", "min_", "performance_", "completion_", "tasks_above")):
                mock_stats[f"mock_env/{key}"] = value

        # Store for epoch-level logging
        self._last_stats = mock_stats

    def on_epoch_end(self, epoch: int) -> None:
        """Log mock environment stats at the end of each epoch."""
        if self._last_stats and self._wandb_run:
            # Add epoch info
            payload = {"mock_env/epoch": epoch, **self._last_stats}

            # Log to wandb
            self._wandb_run.log(payload, step=self.context.agent_step)

            # Clear stats after logging
            self._last_stats = {}


def extract_mock_env_metrics(infos: list[dict[str, Any]]) -> Dict[str, float]:
    """Extract mock environment metrics from info dictionaries for wandb logging."""
    if not infos:
        return {}

    # Get the most recent info dict
    latest_info = infos[-1]

    metrics = {}

    # Extract task-specific metrics
    for key, value in latest_info.items():
        if isinstance(value, (int, float)):
            if key.startswith(("task_", "mean_", "max_", "min_")):
                metrics[f"mock_env/{key}"] = float(value)
            elif key in ["epoch", "tasks_above_threshold", "total_samples_this_epoch"]:
                metrics[f"mock_env/{key}"] = float(value)

    # Extract aggregate metrics from lists
    for list_key in ["task_performances", "task_completion_probs", "sample_counts"]:
        if list_key in latest_info and isinstance(latest_info[list_key], list):
            values = latest_info[list_key]
            if values:
                metrics[f"mock_env/{list_key}_mean"] = float(sum(values) / len(values))
                metrics[f"mock_env/{list_key}_max"] = float(max(values))
                metrics[f"mock_env/{list_key}_min"] = float(min(values))

                # Log individual values for first few tasks
                for i, val in enumerate(values[:5]):  # Limit to first 5 tasks
                    metrics[f"mock_env/{list_key}_{i}"] = float(val)

    return metrics
