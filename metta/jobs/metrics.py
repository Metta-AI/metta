"""WandB metrics extraction utilities.

Consolidated metrics extraction for:
- Extracting wandb run info from logs
- Fetching metrics from wandb API
- Extracting checkpoint paths
- Extracting job IDs
"""

import re
from dataclasses import dataclass
from typing import Optional

import wandb


@dataclass
class WandBInfo:
    """WandB run information extracted from logs."""

    project: str
    entity: str
    run_id: str
    run_name: str
    url: str

    @property
    def run_path(self) -> str:
        """Get wandb run path (entity/project/run_id)."""
        return f"{self.entity}/{self.project}/{self.run_id}"


def extract_wandb_info(log_text: str) -> Optional[WandBInfo]:
    """Extract WandB run information from job logs.

    Looks for patterns like:
        wandb: View run at https://wandb.ai/entity/project/runs/run_id
        wandb: Run data is saved locally in /path/to/wandb/run-timestamp-run_id
        wandb: View project at https://wandb.ai/entity/project

    Args:
        log_text: Full log text from job

    Returns:
        WandBInfo if found, None otherwise
    """
    # Pattern 1: Direct run URL
    # Example: "wandb: View run at https://wandb.ai/metta-ai/metta/runs/abc123xyz"
    run_url_pattern = r"wandb:.*?View run.*?https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s]+)"
    match = re.search(run_url_pattern, log_text)

    if match:
        entity = match.group(1)
        project = match.group(2)
        run_id = match.group(3)

        # Try to extract run name from subsequent lines
        # Pattern: "wandb: Run name: my_run_name"
        run_name_pattern = r"wandb:.*?Run name:\s*([^\n]+)"
        name_match = re.search(run_name_pattern, log_text)
        run_name = name_match.group(1).strip() if name_match else run_id

        url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

        return WandBInfo(
            project=project,
            entity=entity,
            run_id=run_id,
            run_name=run_name,
            url=url,
        )

    # Pattern 2: Local run directory
    # Example: "wandb: Run data is saved locally in /path/wandb/run-20240115_143022-abc123xyz"
    local_dir_pattern = r"wandb:.*?Run data is saved locally.*?run-[^-]+-([^\s/]+)"
    match = re.search(local_dir_pattern, log_text)

    if match:
        run_id = match.group(1)

        # Need to extract entity/project from project URL
        # Example: "wandb: View project at https://wandb.ai/entity/project"
        project_url_pattern = r"wandb:.*?View project.*?https://wandb\.ai/([^/]+)/([^\s]+)"
        project_match = re.search(project_url_pattern, log_text)

        if project_match:
            entity = project_match.group(1)
            project = project_match.group(2)

            # Try to extract run name
            run_name_pattern = r"wandb:.*?Run name:\s*([^\n]+)"
            name_match = re.search(run_name_pattern, log_text)
            run_name = name_match.group(1).strip() if name_match else run_id

            url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

            return WandBInfo(
                project=project,
                entity=entity,
                run_id=run_id,
                run_name=run_name,
                url=url,
            )

    return None


def extract_final_metrics(log_text: str) -> dict[str, float]:
    """Extract final metrics from training logs.

    Looks for common patterns in training output like:
        Final episode_reward: 12.34
        Final train/loss: 0.0123
        episode_length: 456

    Args:
        log_text: Full log text from job

    Returns:
        Dict of metric name -> value
    """
    metrics = {}

    # Pattern: "metric_name: value" or "metric_name = value"
    # Look for lines with numeric values
    patterns = [
        r"Final\s+([a-zA-Z_][a-zA-Z0-9_/]*)\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
        r"([a-zA-Z_][a-zA-Z0-9_/]*)\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, log_text):
            name = match.group(1).strip()
            value_str = match.group(2)

            try:
                value = float(value_str)
                # Only keep if reasonable metric name
                if len(name) >= 3 and not name.startswith("_"):
                    metrics[name] = value
            except ValueError:
                continue

    return metrics


def extract_checkpoint_path(log_text: str) -> Optional[str]:
    """Extract checkpoint save path from training logs.

    Looks for patterns like:
        Saved checkpoint to: /path/to/checkpoint.pt
        Checkpoint saved: s3://bucket/path/checkpoint.pt

    Args:
        log_text: Full log text from job

    Returns:
        Checkpoint path if found, None otherwise
    """
    patterns = [
        r"Saved checkpoint to:\s*([^\s\n]+)",
        r"Checkpoint saved:\s*([^\s\n]+)",
        r"Writing checkpoint.*?:\s*([^\s\n]+)",
    ]

    # Search from end of log (most recent checkpoint)
    lines = log_text.split("\n")
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

    return None


def extract_skypilot_job_id(log_text: str) -> Optional[str]:
    """Extract SkyPilot job ID from launcher logs.

    Looks for patterns like:
        Job submitted with ID: 123
        Submitted job 456
        Job ID: 789

    Args:
        log_text: Full log text from launcher

    Returns:
        Job ID if found, None otherwise
    """
    patterns = [
        r"Job submitted with ID:\s*(\d+)",
        r"Submitted job\s*(\d+)",
        r"Job ID:\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return match.group(1)

    return None


def fetch_wandb_metrics(
    wandb_info: WandBInfo,
    metric_keys: list[str],
    last_n_percent: float = 0.25,
) -> dict[str, float]:
    """Fetch metrics from wandb API and average over last N% of samples.

    Args:
        wandb_info: WandB run information
        metric_keys: List of metric keys to fetch
        last_n_percent: Fraction of samples to average (default: 0.25 = last 25%)

    Returns:
        Dict of metric_key -> averaged value
    """
    metrics: dict[str, float] = {}

    try:
        api = wandb.Api()
        run = api.run(wandb_info.run_path)

        for metric_key in metric_keys:
            try:
                # Fetch history for this metric
                history = run.history(keys=[metric_key], pandas=False)

                if not history:
                    print(f"     Warning: No history found for metric {metric_key}")
                    continue

                # Extract values
                values = [row.get(metric_key) for row in history if metric_key in row and row[metric_key] is not None]

                if not values:
                    print(f"     Warning: No values found for metric {metric_key}")
                    continue

                # Calculate average over last N%
                n_samples = max(1, int(len(values) * last_n_percent))
                last_values = values[-n_samples:]
                avg = sum(last_values) / len(last_values)

                print(f"     Wandb metric {metric_key}: {avg:.2f} (avg of last {len(last_values)} samples)")
                metrics[metric_key] = avg

            except Exception as e:
                print(f"     Warning: Failed to fetch metric {metric_key}: {e}")
                continue

    except Exception as e:
        print(f"     Error fetching from wandb: {e}")

    return metrics
