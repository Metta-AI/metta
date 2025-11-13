"""Simple log parsing and WandB metrics fetching for job monitoring."""

import ast
import json
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


def _strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes."""
    # Pattern matches ANSI escape sequences: \x1b[...m
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[mGKHJh]")
    return ansi_pattern.sub("", text)


def parse_cogames_stats_from_logs(
    log_text: str, metric_keys: list[str], last_n_percent: float = 0.25
) -> dict[str, dict[str, float]]:
    """Parse cogames training stats from log output and average over last N%.

    Args:
        log_text: Complete log output from cogames train --log-outputs
        metric_keys: List of metric names to extract (e.g., ["SPS", "agent_steps"])
        last_n_percent: Fraction of samples to average over (default: 0.25 = last 25%)

    Returns:
        Dictionary mapping metric names to dicts with 'value' and 'count' keys
        Example: {"SPS": {"value": 42000.0, "count": 10}}

    The --log-outputs format produces single-line dicts after "Training:" or "Evaluation:" markers.
    Format: "Training: 2025-10-30 17:14:12.212417+00:00 {'SPS': 35000.0, ...}"
    """
    metrics: dict[str, dict[str, float]] = {}

    # Store all values for each metric across all log entries
    all_values: dict[str, list[float]] = {key: [] for key in metric_keys}

    # Strip ANSI codes first
    clean_text = _strip_ansi_codes(log_text)

    # Parse single-line format: "Training: <timestamp> {dict}" or "Evaluation: <timestamp> {dict}"
    # Pattern matches lines starting with Training: or Evaluation: followed by timestamp and dict
    pattern = re.compile(r"(?:Training|Evaluation):\s+[\d\-:+.\s]+\s+(\{.*\})")

    for match in pattern.finditer(clean_text):
        dict_str = match.group(1)
        try:
            # Remove numpy type wrappers like np.float64(), np.int64(), etc.
            # These appear in Training blocks from pufferlib's mean_and_log()
            dict_str = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", dict_str)

            stats_dict = ast.literal_eval(dict_str)

            # Extract requested metrics
            for key in metric_keys:
                if key in stats_dict:
                    value = stats_dict[key]
                    if value is not None:
                        try:
                            # Handle both single values and lists
                            if isinstance(value, list):
                                # For lists, take the mean
                                if value:  # Non-empty list
                                    numeric_values = [float(v) for v in value]
                                    all_values[key].append(sum(numeric_values) / len(numeric_values))
                            else:
                                # Single value
                                all_values[key].append(float(value))
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Skipping non-numeric value for {key}: {value} ({e})")

        except (SyntaxError, ValueError) as e:
            logger.debug(f"Failed to parse stats dict: {e}")
            continue

    # Compute averages over last N% of samples
    for key, values in all_values.items():
        if not values:
            logger.debug(f"No values found for metric {key}")
            continue

        # Average over last N%
        n_samples = max(1, int(len(values) * last_n_percent))
        last_values = values[-n_samples:]
        avg = sum(last_values) / len(last_values)

        logger.info(f"Cogames metric {key}: {avg:.2f} (avg of last {len(last_values)}/{len(values)} samples)")
        metrics[key] = {"value": avg, "count": len(last_values)}

    return metrics


def parse_cogames_eval_results(log_text: str, metric_keys: list[str]) -> dict[str, dict[str, float]]:
    """Parse cogames evaluation results from JSON output.

    The eval output format from cogames_train_eval.py:
        EvalResults: 2025-10-30 12:34:56.789+00:00 {"generated_at": "...", "missions": [...]}

    Args:
        log_text: Complete log output containing EvalResults line
        metric_keys: List of metric names to extract from eval results
            Can be paths like "avg_game_stats.food_spawned" or "avg_agent_metrics.food_consumed"

    Returns:
        Dictionary mapping metric names to dicts with 'value' and 'count' keys
        Example: {"avg_game_stats.food_spawned": {"value": 42.0, "count": 1}}
    """
    metrics: dict[str, dict[str, float]] = {}

    # Strip ANSI codes first
    clean_text = _strip_ansi_codes(log_text)

    # Parse eval results - format is "EvalResults: <timestamp> <text+json>"
    # The JSON starts with { and may be multiline, possibly with text before it
    pattern = re.compile(r"EvalResults:\s+[\d\-:+.\s]+\s+(.*)", re.DOTALL)
    match = pattern.search(clean_text)

    if not match:
        logger.warning("No EvalResults found in log output")
        return metrics

    # Extract everything after "EvalResults: timestamp"
    content = match.group(1)

    # Find the JSON object within the content (starts with { and ends with })
    # This handles cases where there's text before the JSON
    json_start = content.find("{")
    if json_start == -1:
        logger.warning("No JSON object found in EvalResults")
        return metrics

    json_str = content[json_start:]
    try:
        eval_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse eval JSON: {e}")
        return metrics

    # Extract missions data
    missions = eval_data.get("missions", [])
    if not missions:
        logger.warning("No missions found in eval results")
        return metrics

    # For now, take the first mission (most common case is single mission)
    mission = missions[0]

    # Build a flat dict of available metrics
    available_metrics = {}

    # Add avg_game_stats metrics
    for key, value in mission.get("avg_game_stats", {}).items():
        available_metrics[f"avg_game_stats.{key}"] = value

    # Add policy-level metrics (take first policy if multiple)
    policy_summaries = mission.get("policy_summaries", [])
    if policy_summaries:
        policy = policy_summaries[0]
        for key, value in policy.get("avg_agent_metrics", {}).items():
            available_metrics[f"avg_agent_metrics.{key}"] = value

    # Extract requested metrics
    for key in metric_keys:
        if key in available_metrics:
            try:
                value = float(available_metrics[key])
                metrics[key] = {"value": value, "count": 1}
                logger.info(f"Cogames eval metric {key}: {value:.2f}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert {key}={available_metrics[key]} to float: {e}")
        else:
            logger.warning(f"Metric {key} not found in eval results. Available: {list(available_metrics.keys())}")

    return metrics


def extract_cogames_run_name(log_text: str) -> Optional[str]:
    """Extract cogames run name from log output.

    Cogames doesn't use a 'run=' parameter like metta tools, so we need to
    look for other identifiers. This is a placeholder for future implementation.

    Args:
        log_text: Complete log output

    Returns:
        Run name if found, None otherwise
    """
    # TODO: Implement if cogames adds run name to logs
    # For now, we can use the job name as the identifier
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


def fetch_cogames_metrics_from_logs(
    log_text: str,
    metric_keys: list[str],
) -> dict[str, float] | None:
    """Parse cogames metrics from log output.

    Uses cogames_metrics.parse_cogames_stats_from_logs to extract metrics.

    Args:
        log_text: Complete log output from cogames_train_eval.py
        metric_keys: List of metric names to extract (e.g., ["reward", "SPS"])

    Returns:
        Dictionary mapping metric names to values.
        Returns None if no metrics were extracted.
        Example: {"reward": 15.3, "SPS": 1200.0}
    """
    metrics_data = parse_cogames_stats_from_logs(log_text, metric_keys)

    if not metrics_data:
        return None

    # Extract values from {metric: {"value": X, "count": Y}} format
    metrics_values = {key: data["value"] for key, data in metrics_data.items()}

    return metrics_values if metrics_values else None
