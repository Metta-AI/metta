from __future__ import annotations

from tensordict import TensorDict
from torch import Tensor

from metta.rl.training.experience import Experience


def sequential_sample(buffer: Experience, mb_idx: int) -> tuple[TensorDict, Tensor]:
    """Wrapper for Experience.sample_sequential() - for backward compatibility."""
    return buffer.sample_sequential(mb_idx)


def prio_sample(
    buffer: Experience,
    mb_idx: int,
    epoch: int,
    total_timesteps: int,
    batch_size: int,
    prio_alpha: float,
    prio_beta0: float,
    advantages: Tensor,
) -> tuple[TensorDict, Tensor, Tensor]:
    """Wrapper for Experience.sample_prioritized() - for backward compatibility."""
    return buffer.sample_prioritized(
        mb_idx,
        epoch,
        total_timesteps,
        batch_size,
        prio_alpha,
        prio_beta0,
        advantages,
    )
