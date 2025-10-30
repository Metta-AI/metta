"""Parse and track cogames training stats from --log-outputs."""

import ast
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes and Rich formatting."""
    # Pattern matches ANSI escape sequences and hyperlinks
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[mGKHJh]|\x1b]8;[^\x1b]*\x1b\\|\[2m|\[0m")
    text = ansi_pattern.sub("", text)

    # Also remove file references like "train.py:337" that appear after values
    # These show up even after ANSI stripping from Rich console.log() hyperlinks
    file_ref_pattern = re.compile(r"\s+[a-z_]+\.py:\d+")
    text = file_ref_pattern.sub("", text)

    return text


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

    The --log-outputs format produces multi-line dicts after "Training:" markers.
    """
    metrics: dict[str, dict[str, float]] = {}

    # Store all values for each metric across all log entries
    all_values: dict[str, list[float]] = {key: [] for key in metric_keys}

    # Strip ANSI codes first
    clean_text = _strip_ansi_codes(log_text)

    # Extract multi-line dicts: collect lines between "Training:" markers
    training_blocks = []
    current_block = []
    in_block = False

    for line in clean_text.splitlines():
        if "Training:" in line:
            if current_block:
                training_blocks.append("\n".join(current_block))
            current_block = []
            in_block = True
        elif in_block:
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        training_blocks.append("\n".join(current_block))

    # Parse each training block to extract the dict
    for block in training_blocks:
        try:
            # Find the dict portion - starts with '{' and ends with '}'
            start_idx = block.find("{")
            end_idx = block.rfind("}")
            if start_idx != -1 and end_idx != -1:
                dict_str = block[start_idx : end_idx + 1]
                stats_dict = ast.literal_eval(dict_str)

                # Extract requested metrics
                for key in metric_keys:
                    if key in stats_dict:
                        value = stats_dict[key]
                        if value is not None:
                            try:
                                all_values[key].append(float(value))
                            except (ValueError, TypeError):
                                logger.debug(f"Skipping non-numeric value for {key}: {value}")

        except (SyntaxError, ValueError) as e:
            logger.debug(f"Failed to parse training block: {e}")
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
