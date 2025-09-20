"""Rich console progress display for training."""

import os
import sys

from rich.console import Console


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


def log_rich_progress(
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    steps_per_sec: float,
    train_pct: float,
    rollout_pct: float,
    stats_pct: float,
    run_name: str | None = None,
    heart_value: float | None = None,
    heart_rate: float | None = None,
) -> None:
    """Log training progress using a compact rich-formatted line."""
    console = Console()

    progress_pct = (agent_step / total_timesteps) * 100 if total_timesteps > 0 else 0

    segments = []
    if run_name:
        segments.append(f"[bold cyan]{run_name}[/bold cyan]")
    segments.append(f"epoch {epoch}")
    segments.append(
        f"{agent_step:,}/{total_timesteps:,} ({progress_pct:.1f}%)" if total_timesteps else f"{agent_step:,}"
    )
    segments.append(f"[yellow]{steps_per_sec:,.0f} sps[/yellow]")
    segments.append(f"train {train_pct:.0f}% • rollout {rollout_pct:.0f}% • stats {stats_pct:.0f}%")
    if heart_value is not None:
        heart_segment = f"heart.get {heart_value:.3f}"
        if heart_rate is not None:
            heart_segment += f" ({heart_rate:.3f}/s)"
        segments.append(heart_segment)

    console.print(" • ".join(segments))
