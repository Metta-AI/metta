"""Training loop helper functions."""

import logging

from metta.utils.progress import log_rich_progress, should_use_rich_console

logger = logging.getLogger(__name__)


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
    run_name: str,
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

        run_info = f" [{run_name}]" if run_name else ""

        logger.info(
            f"Epoch {epoch}{run_info} / "
            f"{human_readable_si(steps_per_sec, 'sps')} / "
            f"{agent_step / total_timesteps:.2%} of {human_readable_si(total_timesteps, 'steps')} / "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
        )
