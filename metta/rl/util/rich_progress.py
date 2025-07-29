"""Rich console progress display for training."""

import os

from rich.console import Console
from rich.table import Table


def should_use_rich_console() -> bool:
    """Determine if rich console output is appropriate.

    Returns:
        False for batch jobs, wandb, skypilot, or when disabled.
        True for interactive terminals.
    """
    # Check if explicitly disabled
    if os.environ.get("DISABLE_RICH_LOGGING", "").lower() in ("1", "true", "yes"):
        return False

    # Check for batch job environments
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID"):
        return False

    # Check for wandb
    if os.environ.get("WANDB_RUN_ID"):
        return False

    # Check for skypilot
    if os.environ.get("SKYPILOT_TASK_ID"):
        return False

    # Check if we have a TTY
    import sys

    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    return True


def create_progress_table(epoch: int) -> Table:
    """Create a rich table for training progress.

    Args:
        epoch: Current training epoch

    Returns:
        Configured Rich table
    """
    table = Table(
        title=f"[bold cyan]Training Progress - Epoch {epoch}[/bold cyan]",
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
) -> None:
    """Log training progress using rich console tables.

    Args:
        epoch: Current epoch
        agent_step: Current agent step
        total_timesteps: Total timesteps to train
        steps_per_sec: Steps per second
        train_pct: Percentage of time in training
        rollout_pct: Percentage of time in rollout
        stats_pct: Percentage of time in stats processing
    """
    console = Console()
    table = create_progress_table(epoch)

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
