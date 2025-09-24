"""Progress logging utilities and component."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table

from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


def should_use_rich_console() -> bool:
    """Determine if rich console output is appropriate based on terminal context."""
    if os.environ.get("DISABLE_RICH_LOGGING", "").lower() in ("1", "true", "yes"):
        return False

    if any(os.environ.get(var) for var in ["SLURM_JOB_ID", "PBS_JOBID", "WANDB_RUN_ID", "SKYPILOT_TASK_ID"]):
        return False

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

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metrics", style="cyan", justify="left")
    table.add_column("Progress", style="green", justify="right")
    table.add_column("Values", style="yellow", justify="left")
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
    sps_display = f"{steps_per_sec:,.0f} SPS"
    heart_display = ""
    if heart_value is not None:
        heart_display = f"heart.get {heart_value:.3f}"
        if heart_rate is not None:
            heart_display += f" ({heart_rate:.3f}/s)"

    table.add_row(
        "Steps",
        f"{agent_step:,} / {total_steps_str} ({progress_pct:.1f}%)",
        sps_display,
    )

    table.add_row(
        "Time",
        f"Train: {train_pct:.0f}% | Rollout: {rollout_pct:.0f}% | Stats: {stats_pct:.0f}%",
        heart_display,
    )

    console.print(table)


def log_training_progress(
    *,
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    prev_agent_step: int,
    train_time: float,
    rollout_time: float,
    stats_time: float,
    run_name: str | None,
    metrics: Dict[str, float] | None,
) -> None:
    """Log training progress with timing breakdown and optional metrics."""

    total_time = train_time + rollout_time + stats_time
    if total_time > 0:
        steps_per_sec = (agent_step - prev_agent_step) / total_time
        train_pct = (train_time / total_time) * 100
        rollout_pct = (rollout_time / total_time) * 100
        stats_pct = (stats_time / total_time) * 100
    else:
        steps_per_sec = train_pct = rollout_pct = stats_pct = 0.0

    heart_value = None
    heart_rate = None
    if metrics:
        heart_value = metrics.get("env_agent/heart.get") or metrics.get("overview/heart.get")
        heart_rate = metrics.get("env_agent/heart.get.rate")

    if should_use_rich_console():
        log_rich_progress(
            epoch=epoch,
            agent_step=agent_step,
            total_timesteps=total_timesteps,
            steps_per_sec=steps_per_sec,
            train_pct=train_pct,
            rollout_pct=rollout_pct,
            stats_pct=stats_pct,
            run_name=run_name,
            heart_value=heart_value,
            heart_rate=heart_rate,
        )
    else:
        label = run_name if run_name else "training"
        if total_timesteps > 0:
            progress_str = f"{agent_step:,}/{total_timesteps:,} ({agent_step / total_timesteps:.1%})"
        else:
            progress_str = f"{agent_step:,}"

        message = (
            f"{label} _ epoch {epoch} _ {progress_str} _ "
            f"{_human_readable_si(steps_per_sec, 'sps')} _ "
            f"train {train_pct:.0f}% _ rollout {rollout_pct:.0f}% _ stats {stats_pct:.0f}%"
        )
        if heart_value is not None:
            segment = f"heart.get {heart_value:.3f}"
            if heart_rate is not None:
                segment += f" ({heart_rate:.3f}/s)"
            message = f"{message} _ {segment}"
        logger.info(message)


def _human_readable_si(value: float, unit: str = "") -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} G{unit}"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M{unit}"
    if value >= 1_000:
        return f"{value / 1_000:.2f} k{unit}"
    return f"{value:.0f} {unit}" if unit else f"{value:.0f}"


class ProgressLogger(TrainerComponent):
    """Master-only component that logs epoch progress."""

    _master_only = True

    def __init__(self) -> None:
        super().__init__(epoch_interval=1)
        self._previous_agent_step: int = 0

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self._previous_agent_step = context.agent_step

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        ctx = self.context
        metrics = self._latest_metrics()
        log_training_progress(
            epoch=ctx.epoch,
            agent_step=ctx.agent_step,
            prev_agent_step=self._previous_agent_step,
            total_timesteps=ctx.config.total_timesteps,
            train_time=ctx.stopwatch.get_last_elapsed("_train"),
            rollout_time=ctx.stopwatch.get_last_elapsed("_rollout"),
            stats_time=ctx.stopwatch.get_last_elapsed("_process_stats"),
            run_name=ctx.run_name,
            metrics=metrics,
        )
        self._previous_agent_step = ctx.agent_step

    def _latest_metrics(self) -> Optional[Dict[str, float]]:
        stats_reporter = getattr(self.context, "stats_reporter", None)
        if stats_reporter is None:
            return None
        latest = getattr(stats_reporter, "get_latest_payload", None)
        return latest() if latest else None
