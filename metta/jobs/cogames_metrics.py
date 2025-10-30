"""Parse and track cogames training stats from --log-outputs."""

import ast
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


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
