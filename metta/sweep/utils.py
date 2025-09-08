"""Utility functions for sweep orchestration."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.sweep.models import RunInfo

logger = logging.getLogger(__name__)


def make_monitor_table(
    runs: list["RunInfo"],
    title: str = "Run Status Table",
    logger_prefix: str = "",
    include_score: bool = True,
    truncate_run_id: bool = True,
) -> list[str]:
    """Create a formatted table showing run status.

    Args:
        runs: List of RunInfo objects to display
        title: Title for the table
        logger_prefix: Prefix to add to each log line (e.g., "[OptimizingScheduler]")
        include_score: Whether to include the score column
        truncate_run_id: Whether to truncate run IDs to just show trial numbers

    Returns:
        List of formatted lines that can be logged
    """
    lines = []
    prefix = f"{logger_prefix} " if logger_prefix else ""

    # Title
    lines.append(f"{prefix}{title}:")
    lines.append(f"{prefix}{'=' * 100}")

    # Header
    if include_score:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30} {'Score':<15}")
    else:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30}")
    lines.append(f"{prefix}{'-' * 100}")

    # Rows
    for run in runs:
        # Format run ID
        display_id = run.run_id
        if truncate_run_id and "_trial_" in run.run_id:
            display_id = run.run_id.split("_trial_")[-1]
            display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id

        # Format progress
        if run.total_timesteps and run.current_steps is not None:
            progress_pct = (run.current_steps / run.total_timesteps) * 100
            progress_str = f"{run.current_steps:,}/{run.total_timesteps:,} ({progress_pct:.1f}%)"
        elif run.current_steps is not None:
            progress_str = f"{run.current_steps:,}/?"
        else:
            progress_str = "-"

        # Format score
        if include_score:
            score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30} {score_str:<15}")
        else:
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30}")

    lines.append(f"{prefix}{'=' * 100}")

    return lines
