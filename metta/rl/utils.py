"""Training utilities for Metta RL."""

import logging

import torch
from tensordict import TensorDict

from metta.utils.progress import log_rich_progress, should_use_rich_console

logger = logging.getLogger(__name__)


def ensure_sequence_metadata(td: TensorDict, *, batch_size: int, time_steps: int) -> None:
    """Attach required sequence metadata to ``td`` if missing."""

    total = batch_size * time_steps
    device = td.device
    if "batch" not in td.keys():
        td.set("batch", torch.full((total,), batch_size, dtype=torch.long, device=device))
    if "bptt" not in td.keys():
        td.set("bptt", torch.full((total,), time_steps, dtype=torch.long, device=device))


def should_run(
    epoch: int,
    interval: int,
    *,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval. It is assumed this is only called on master."""
    if not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


def log_training_progress(
    *,
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    prev_agent_step: int,
    train_time: float,
    rollout_time: float,
    stats_time: float,
    run_name: str | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Log training progress with timing breakdown and performance metrics."""
    total_time = train_time + rollout_time + stats_time
    if total_time > 0:
        steps_per_sec = (agent_step - prev_agent_step) / total_time
        train_pct = (train_time / total_time) * 100
        rollout_pct = (rollout_time / total_time) * 100
        stats_pct = (stats_time / total_time) * 100
    else:
        steps_per_sec = train_pct = rollout_pct = stats_pct = 0

    heart_value = None
    heart_rate = None
    if metrics:
        heart_value = metrics.get("env_agent/heart.get") or metrics.get("overview/heart.get")
        heart_rate = metrics.get("env_agent/heart.get.rate")

    # Use rich console if appropriate
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

        def human_readable_si(n: float, unit: str = "") -> str:
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.2f} G{unit}"
            elif n >= 1_000_000:
                return f"{n / 1_000_000:.2f} M{unit}"
            elif n >= 1_000:
                return f"{n / 1_000:.2f} k{unit}"
            return f"{n:.0f} {unit}" if unit else f"{n:.0f}"

        label = run_name if run_name else "training"
        progress_str = (
            f"{agent_step:,}/{total_timesteps:,} ({agent_step / total_timesteps:.1%})"
            if total_timesteps > 0
            else f"{agent_step:,}"
        )
        message = (
            f"{label} _ epoch {epoch} _ {progress_str} _ {human_readable_si(steps_per_sec, 'sps')} _ "
            f"train {train_pct:.0f}% _ rollout {rollout_pct:.0f}% _ stats {stats_pct:.0f}%"
        )
        if heart_value is not None:
            segment = f"heart.get {heart_value:.3f}"
            if heart_rate is not None:
                segment += f" ({heart_rate:.3f}/s)"
            message = f"{message} _ {segment}"
        logger.info(message)
