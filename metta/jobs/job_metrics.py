"""Simple log parsing and WandB metrics fetching for job monitoring."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


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


def parse_run_name(args: list[str]) -> str | None:
    """Extract run name from job config args.

    Parses args list looking for 'run=X' format.
    """
    for arg in args:
        if arg.startswith("run="):
            try:
                return arg.split("=", 1)[1]
            except IndexError:
                continue
    return None


def parse_total_timesteps(args: list[str]) -> int | None:
    """Extract total_timesteps from job config args.

    Parses args list looking for 'trainer.total_timesteps=X' format.
    This is metta training-specific logic.
    """
    for arg in args:
        if arg.startswith("trainer.total_timesteps="):
            try:
                value = arg.split("=", 1)[1]
                return int(value)
            except (ValueError, IndexError):
                continue
    return None


def fetch_wandb_metrics(
    entity: str,
    project: str,
    run_name: str,
    metric_keys: list[str],
    last_n_percent: float = 0.25,
) -> tuple[dict[str, dict[str, float]], int | None]:
    """Fetch metrics from WandB API and average over last N% of samples.

    Args:
        entity: WandB entity (e.g., "metta-research")
        project: WandB project (e.g., "metta")
        run_name: WandB run name (e.g., "job_v2025.10.28-0637_arena_multi_gpu_2b_20251028_063808")
        metric_keys: List of metric names to fetch
        last_n_percent: Fraction of samples to average over (default: 0.25 = last 25%)

    Returns:
        Tuple of (metrics_dict, current_step):
        - metrics_dict: Dictionary mapping metric names to dicts with 'value' and 'count' keys
        - current_step: Current training step (metric/agent_step), or None if not available
        Example: ({"overview/sps": {"value": 42000.0, "count": 100}}, 50000)
    """
    metrics: dict[str, dict[str, float]] = {}
    current_step: int | None = None

    try:
        # import here to delay heavy import until needed
        import wandb

        api = wandb.Api()
        # Find run by name
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        run = next(iter(runs), None)

        if not run:
            logger.warning(f"WandB run not found: {run_name}")
            return metrics, current_step

        for metric_key in metric_keys:
            try:
                history = run.history(keys=[metric_key], pandas=False)

                if not history:
                    logger.debug(f"No history found for metric {metric_key}")
                    continue

                values = [row.get(metric_key) for row in history if metric_key in row and row[metric_key] is not None]

                if not values:
                    logger.debug(f"No values found for metric {metric_key}")
                    continue

                # Average over last N%
                n_samples = max(1, int(len(values) * last_n_percent))
                last_values = values[-n_samples:]
                avg = sum(last_values) / len(last_values)

                logger.info(f"WandB metric {metric_key}: {avg:.2f} (avg of last {len(last_values)} samples)")
                metrics[metric_key] = {"value": avg, "count": len(last_values)}

            except Exception as e:
                logger.warning(f"Failed to fetch metric {metric_key}: {e}")
                continue

        # Always fetch current training step
        try:
            step_history = run.history(keys=["metric/agent_step"], pandas=False)
            if step_history:
                step_values = [
                    row.get("metric/agent_step")
                    for row in step_history
                    if "metric/agent_step" in row and row["metric/agent_step"] is not None
                ]
                if step_values:
                    current_step = int(step_values[-1])  # Get latest step
                    logger.info(f"Current training step: {current_step}")
        except Exception as e:
            logger.debug(f"Failed to fetch current step: {e}")

    except Exception as e:
        logger.warning(f"Error fetching from WandB: {e}")

    return metrics, current_step


def fetch_job_metrics(
    entity: str,
    project: str,
    run_name: str,
    metric_keys: list[str],
    total_timesteps: int | None = None,
) -> dict[str, float] | None:
    """Fetch metrics from WandB and build final metrics dict with optional progress.

    This is a higher-level convenience function that:
    1. Fetches metrics from WandB
    2. Extracts the 'value' from each metric
    3. Adds '_progress' dict if training step info is available

    Args:
        entity: WandB entity
        project: WandB project
        run_name: WandB run name
        metric_keys: List of metric names to fetch
        total_timesteps: Optional total timesteps for progress calculation

    Returns:
        Dictionary mapping metric names to values, with optional '_progress' dict.
        Returns None if no metrics were fetched.
        Example: {"overview/sps": 42000.0, "_progress": {"current_step": 50000, "total_steps": 100000}}
    """
    metrics_data, current_step = fetch_wandb_metrics(
        entity=entity,
        project=project,
        run_name=run_name,
        metric_keys=metric_keys,
    )

    if not metrics_data:
        return None

    # Extract values from {metric: {"value": X, "count": Y}} format
    metrics_values = {key: data["value"] for key, data in metrics_data.items()}

    # Add progress if we have both current step and total steps
    if current_step is not None and total_timesteps is not None:
        metrics_values["_progress"] = {
            "current_step": current_step,
            "total_steps": total_timesteps,
        }

    return metrics_values
