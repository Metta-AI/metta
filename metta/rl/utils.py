"""Training utilities for Metta RL."""

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from metta.utils.progress import log_rich_progress, should_use_rich_console

logger = logging.getLogger(__name__)

BatchInfo = Tuple[int, int]


def ensure_sequence_metadata(td: TensorDict, *, batch_size: int, time_steps: int) -> None:
    """Attach required sequence metadata to ``td`` if missing."""

    total = batch_size * time_steps
    device = td.device
    if "batch" not in td.keys():
        td.set("batch", torch.full((total,), batch_size, dtype=torch.long, device=device))
    if "bptt" not in td.keys():
        td.set("bptt", torch.full((total,), time_steps, dtype=torch.long, device=device))


def flatten_td_for_policy(
    td: TensorDict, action: Optional[torch.Tensor] = None
) -> tuple[TensorDict, Optional[torch.Tensor], BatchInfo]:
    """Flatten tensor dict (and optionally actions) for policy forward passes."""

    batch_shape = tuple(int(dim) for dim in td.batch_size)
    if not batch_shape:
        raise ValueError("TensorDict must have at least one batch dimension")

    batch_size = batch_shape[0]
    time_steps = batch_shape[1] if len(batch_shape) > 1 else 1

    if td.batch_dims > 1:
        td = td.reshape(batch_size * time_steps)

    ensure_sequence_metadata(td, batch_size=batch_size, time_steps=time_steps)

    flat_action = action
    if action is not None and action.dim() == 3:
        flat_action = action.reshape(batch_size * time_steps, action.shape[-1])

    return td, flat_action, (batch_size, time_steps)


def restore_td_from_policy(td: TensorDict, batch_info: BatchInfo) -> TensorDict:
    """Restore TensorDict after a policy forward pass."""

    batch_size, time_steps = batch_info
    if time_steps == 1:
        return td.reshape(batch_size)
    return td.reshape(batch_size, time_steps)


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

        logger.info(
            f"Epoch {epoch} / "
            f"{human_readable_si(steps_per_sec, 'sps')} / "
            f"{agent_step / total_timesteps:.2%} of {human_readable_si(total_timesteps, 'steps')} / "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
        )
