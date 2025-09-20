"""TensorDict shape/metadata helpers shared across RL modules."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def flatten_sequence(td: TensorDict) -> tuple[TensorDict, int, int]:
    """Flatten a [B, T, ...] TensorDict and attach sequence metadata."""

    batch_shape = tuple(int(dim) for dim in td.batch_size)
    if not batch_shape:
        raise ValueError("TensorDict must have at least one batch dimension")

    batch_size = batch_shape[0]
    time_steps = batch_shape[1] if len(batch_shape) > 1 else 1

    if td.batch_dims > 1:
        td = td.reshape(batch_size * time_steps)

    total = batch_size * time_steps
    device = td.device
    td.set("batch", torch.full((total,), batch_size, dtype=torch.long, device=device))
    td.set("bptt", torch.full((total,), time_steps, dtype=torch.long, device=device))

    return td, batch_size, time_steps


def restore_sequence(td: TensorDict, batch_size: int, time_steps: int) -> TensorDict:
    """Restore a flattened TensorDict back to [B, T, ...]."""

    if time_steps == 1:
        return td.reshape(batch_size)
    return td.reshape(batch_size, time_steps)
