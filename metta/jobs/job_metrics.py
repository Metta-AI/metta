"""WandB metrics and artifact extraction from job logs.

Extracts wandb run info, metrics, checkpoints, and job IDs from log output.
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
        return f"{self.entity}/{self.project}/{self.run_id}"


def extract_wandb_info(log_text: str) -> Optional[WandBInfo]:
    """Extract WandB run information from job logs.

    Looks for wandb URLs and local directory paths in log output.
    """
    run_url_pattern = r"wandb:.*?View run.*?https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s]+)"
    match = re.search(run_url_pattern, log_text)

    if match:
        entity = match.group(1)
        project = match.group(2)
        run_id = match.group(3)
        run_name_pattern = r"wandb:.*?Run name:\s*([^\n]+)"
        name_match = re.search(run_name_pattern, log_text)
        run_name = name_match.group(1).strip() if name_match else run_id
        url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        return WandBInfo(project=project, entity=entity, run_id=run_id, run_name=run_name, url=url)
    local_dir_pattern = r"wandb:.*?Run data is saved locally.*?run-[^-]+-([^\s/]+)"
    match = re.search(local_dir_pattern, log_text)

    if match:
        run_id = match.group(1)
        project_url_pattern = r"wandb:.*?View project.*?https://wandb\.ai/([^/]+)/([^\s]+)"
        project_match = re.search(project_url_pattern, log_text)

        if project_match:
            entity = project_match.group(1)
            project = project_match.group(2)
            run_name_pattern = r"wandb:.*?Run name:\s*([^\n]+)"
            name_match = re.search(run_name_pattern, log_text)
            run_name = name_match.group(1).strip() if name_match else run_id
            url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            return WandBInfo(project=project, entity=entity, run_id=run_id, run_name=run_name, url=url)

    return None


def extract_final_metrics(log_text: str) -> dict[str, float]:
    """Extract final metrics from training logs.

    Parses metric_name: value or metric_name = value patterns.
    """
    metrics = {}
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
                if len(name) >= 3 and not name.startswith("_"):
                    metrics[name] = value
            except ValueError:
                continue

    return metrics


def extract_checkpoint_path(log_text: str) -> Optional[str]:
    """Extract checkpoint save path from training logs.

    Returns the most recent checkpoint path found.
    """
    patterns = [
        r"Saved checkpoint to:\s*([^\s\n]+)",
        r"Checkpoint saved:\s*([^\s\n]+)",
        r"Writing checkpoint.*?:\s*([^\s\n]+)",
    ]
    lines = log_text.split("\n")
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

    return None


def extract_skypilot_job_id(log_text: str) -> Optional[str]:
    """Extract SkyPilot job ID from launcher logs."""
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
    """Fetch metrics from wandb API and average over last N% of samples."""
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
