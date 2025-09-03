"""Rich console progress display for training."""

import os
import sys

from rich.console import Console
from rich.table import Table


def should_use_rich_console() -> bool:
    """Determine if rich console output is appropriate based on environment and TTY availability."""
    # Check if explicitly disabled
    if os.environ.get("DISABLE_RICH_LOGGING", "").lower() in ("1", "true", "yes"):
        return False

    # Check for batch job environments, wandb, or skypilot
    if any(os.environ.get(var) for var in ["SLURM_JOB_ID", "PBS_JOBID", "WANDB_RUN_ID", "SKYPILOT_TASK_ID"]):
        return False

    # Check if we have a TTY
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def create_progress_table(epoch: int, run_name: str | None = None) -> Table:
    """Create a configured rich table for displaying training progress."""
    title = f"[bold cyan]Training Progress - Epoch {epoch}"
    if run_name:
        title += f"\n{run_name}"
    title += "[/bold cyan]"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Progress", style="green", justify="right")
    table.add_column("Rate", style="yellow", justify="left")

    return table


def log_rich_progress(
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    steps_per_sec: float,
    train_pct: float,
    rollout_pct: float,
    stats_pct: float,
    run_name: str | None = None,
) -> None:
    """Log training progress using rich console tables with steps and time breakdown."""
    console = Console()
    table = create_progress_table(epoch, run_name)

    # Format total timesteps
    if total_timesteps >= 1e9:
        total_steps_str = f"{total_timesteps:.0e}"
    else:
        total_steps_str = f"{total_timesteps:,}"

    # Add training progress row
    progress_pct = (agent_step / total_timesteps) * 100 if total_timesteps > 0 else 0
    table.add_row(
        "Training Steps",
        f"{agent_step:,} / {total_steps_str} ({progress_pct:.1f}%)",
        f"[dim]{steps_per_sec:.0f} steps/sec[/dim]",
    )

    # Add time breakdown row
    table.add_row(
        "Time Breakdown",
        f"Train: {train_pct:.0f}% | Rollout: {rollout_pct:.0f}% | Stats: {stats_pct:.0f}%",
        "",
    )

    # Print the table
    console.print(table)
