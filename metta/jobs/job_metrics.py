"""Simple log parsing and WandB metrics fetching for job monitoring."""

import re
from typing import Optional

import wandb


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
    entity: str,
    project: str,
    run_name: str,
    metric_keys: list[str],
    last_n_percent: float = 0.25,
) -> dict[str, float]:
    """Fetch metrics from WandB API and average over last N% of samples.

    Args:
        entity: WandB entity (e.g., "metta-research")
        project: WandB project (e.g., "metta")
        run_name: WandB run name (e.g., "job_v2025.10.28-0637_arena_multi_gpu_2b_20251028_063808")
        metric_keys: List of metric names to fetch
        last_n_percent: Fraction of samples to average over (default: 0.25 = last 25%)

    Returns:
        Dictionary mapping metric names to their averaged values
    """
    metrics: dict[str, float] = {}

    try:
        api = wandb.Api()
        # Find run by name
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        run = next(iter(runs), None)

        if not run:
            print(f"     Warning: WandB run not found: {run_name}")
            return metrics

        for metric_key in metric_keys:
            try:
                history = run.history(keys=[metric_key], pandas=False)

                if not history:
                    print(f"     Warning: No history found for metric {metric_key}")
                    continue

                values = [row.get(metric_key) for row in history if metric_key in row and row[metric_key] is not None]

                if not values:
                    print(f"     Warning: No values found for metric {metric_key}")
                    continue

                # Average over last N%
                n_samples = max(1, int(len(values) * last_n_percent))
                last_values = values[-n_samples:]
                avg = sum(last_values) / len(last_values)

                print(f"     WandB metric {metric_key}: {avg:.2f} (avg of last {len(last_values)} samples)")
                metrics[metric_key] = avg

            except Exception as e:
                print(f"     Warning: Failed to fetch metric {metric_key}: {e}")
                continue

    except Exception as e:
        print(f"     Error fetching from WandB: {e}")

    return metrics
