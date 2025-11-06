
import tensordict
import torch

import metta.rl.training.experience


def sample_minibatch_sequential(
    buffer: metta.rl.training.experience.Experience, mb_idx: int
) -> tuple[tensordict.TensorDict, torch.Tensor]:
    """Simple way to sample a contiguous minibatch from the replay buffer in order."""
    segments_per_mb = buffer.minibatch_segments
    total_segments = buffer.segments
    num_minibatches = max(buffer.num_minibatches, 1)

    mb_idx_mod = int(mb_idx % num_minibatches)
    start = mb_idx_mod * segments_per_mb
    end = start + segments_per_mb

    device = buffer.device

    if end <= total_segments:
        idx = torch.arange(start, end, dtype=torch.long, device=device)
    else:
        overflow = end - total_segments
        front = torch.arange(start, total_segments, dtype=torch.long, device=device)
        back = torch.arange(0, overflow, dtype=torch.long, device=device)
        idx = torch.cat((front, back), dim=0)

    minibatch = buffer.buffer[idx]
    return minibatch.clone(), idx
