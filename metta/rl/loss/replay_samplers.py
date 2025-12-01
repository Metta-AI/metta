from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.training.batch import calculate_prioritized_sampling_params
from metta.rl.training.experience import Experience


def sequential_sample(buffer: Experience, mb_idx: int) -> tuple[TensorDict, Tensor]:
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
    if prio_alpha <= 0.0:
        # why you call me if you don't want prio-style sampling?
        minibatch, idx = sequential_sample(buffer, mb_idx)
        return (
            minibatch,
            idx,
            torch.ones((minibatch.shape[0], minibatch.shape[1]), device=buffer.device, dtype=torch.float32),
        )

    anneal_beta = calculate_prioritized_sampling_params(
        epoch=epoch,
        total_timesteps=total_timesteps,
        batch_size=batch_size,
        prio_alpha=prio_alpha,
        prio_beta0=prio_beta0,
    )

    adv_magnitude = advantages.abs().sum(dim=1)
    prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
    prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
    all_prio_is_weights = (buffer.segments * prio_probs) ** -anneal_beta

    # Sample segment indices
    idx = torch.multinomial(prio_probs, buffer.minibatch_segments)

    minibatch = buffer.buffer[idx].clone()

    return minibatch, idx, all_prio_is_weights[idx, None]  # av check
