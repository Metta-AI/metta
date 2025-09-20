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


def _format_total_steps(total_timesteps: int) -> str:
    if total_timesteps <= 0:
        return "?"
    if total_timesteps >= 1_000_000_000:
        return f"{total_timesteps:.0e}"
    return f"{total_timesteps:,}"


def _create_progress_table(epoch: int, run_name: str | None) -> Table:
    if run_name:
        title = f"[bold cyan]{run_name} · Training Progress - Epoch {epoch}[/bold cyan]"
    else:
        title = f"[bold cyan]Training Progress - Epoch {epoch}[/bold cyan]"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Steps", style="cyan", justify="left")
    table.add_column("Progress", style="green", justify="right")
    table.add_column("Time", style="yellow", justify="left")
    return table


def log_rich_progress(
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    steps_per_sec: float,
    train_pct: float,
    rollout_pct: float,
    stats_pct: float,
    run_name: str | None,
    heart_value: float | None,
    heart_rate: float | None,
) -> None:
    """Render training progress in a rich table."""

    console = Console()
    table = _create_progress_table(epoch, run_name)

    total_steps_str = _format_total_steps(total_timesteps)
    progress_pct = (agent_step / total_timesteps) * 100 if total_timesteps > 0 else 0.0
    heart_display = ""
    if heart_value is not None:
        heart_display = f"heart.get {heart_value:.3f}"
        if heart_rate is not None:
            heart_display += f" ({heart_rate:.3f}/s)"

    table.add_row(
        "Training Steps",
        f"{agent_step:,} / {total_steps_str} ({progress_pct:.1f}%)",
        heart_display,
    )

    table.add_row(
        "Time Breakdown",
        f"Train: {train_pct:.0f}% | Rollout: {rollout_pct:.0f}% | Stats: {stats_pct:.0f}%",
        f"[dim]{steps_per_sec:,.0f} steps/sec[/dim]",
    )

    console.print(table)
