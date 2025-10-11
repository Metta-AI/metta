"""Metrics extraction utilities for release validation."""

from __future__ import annotations

import re
from typing import Optional

import wandb

# Regex pattern for extracting wandb URL from logs
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s]+)")


def extract_wandb_run_info(log_text: str) -> Optional[tuple[str, str, str]]:
    """Extract wandb entity, project, and run_id from log text."""
    match = _WANDB_URL_RE.search(log_text)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    return None


def fetch_wandb_metric(
    entity: str, project: str, run_id: str, metric_key: str, last_n_percent: float = 0.25
) -> Optional[float]:
    """Fetch a metric from wandb and return average over last N% of timesteps."""
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        # Fetch history for the metric
        history = run.history(keys=[metric_key], pandas=False)

        if not history:
            print(f"Warning: No history found for metric {metric_key}")
            return None

        # Extract values
        values = [row.get(metric_key) for row in history if metric_key in row and row[metric_key] is not None]

        if not values:
            print(f"Warning: No values found for metric {metric_key}")
            return None

        # Calculate average over last N%
        n_samples = max(1, int(len(values) * last_n_percent))
        last_values = values[-n_samples:]
        avg = sum(last_values) / len(last_values)

        print(f"     Wandb metric {metric_key}: {avg:.2f} (avg of last {len(last_values)} samples)")
        return avg

    except Exception as e:
        print(f"Warning: Failed to fetch wandb metric {metric_key}: {e}")
        return None


def extract_metrics(log_text: str, wandb_metrics: Optional[list[str]] = None) -> dict[str, float]:
    """Extract metrics from wandb for training tasks."""
    metrics: dict[str, float] = {}

    if not wandb_metrics:
        return metrics

    wandb_info = extract_wandb_run_info(log_text)
    if not wandb_info:
        print("     Error: No wandb URL found in logs")
        return metrics

    entity, project, run_id = wandb_info
    print(f"     Fetching {len(wandb_metrics)} metrics from wandb: {entity}/{project}/{run_id}...")
    for metric_key in wandb_metrics:
        value = fetch_wandb_metric(entity, project, run_id, metric_key)
        if value is not None:
            metrics[metric_key] = value

    return metrics
