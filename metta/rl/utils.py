"""Training utilities for Metta RL."""

import tensordict
import torch


def ensure_sequence_metadata(td: tensordict.TensorDict, *, batch_size: int, time_steps: int) -> None:
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
