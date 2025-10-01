"""Shared helpers for constructing auxiliary transformer tokens."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import torch
from tensordict import TensorDict

ZeroFactory = Callable[[tuple[int, ...], torch.device, torch.dtype], torch.Tensor]


def prepare_auxiliary_signals(
    td: TensorDict,
    *,
    batch_size: int,
    time_steps: int,
    action_dim: int,
    device: torch.device,
    reward_dtype: torch.dtype,
    action_dtype: torch.dtype,
    zeros_factory: ZeroFactory,
    reward_keys: Sequence[str] = ("rewards",),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened reward, reset, and previous-action tensors.

    The helper normalizes scalar rewards/done flags and past action tensors so
    callers can project them into transformer token embeddings regardless of
    whether the source ``TensorDict`` used rollout or training keys.
    """

    total = batch_size * time_steps

    reward_tensor = _extract_first_available(td, reward_keys)
    if reward_tensor is None:
        reward = zeros_factory((total, 1), device, reward_dtype)
    else:
        reward = reward_tensor.view(total, -1).to(device=device, dtype=reward_dtype)
        if reward.size(1) != 1:
            reward = reward[:, :1]

    dones = td.get("dones")
    truncateds = td.get("truncateds")
    if dones is None and truncateds is None:
        resets = zeros_factory((total, 1), device, reward_dtype)
    else:
        if dones is None and truncateds is not None:
            dones = torch.zeros_like(truncateds)
        if truncateds is None and dones is not None:
            truncateds = torch.zeros_like(dones)
        assert dones is not None
        assert truncateds is not None
        resets = torch.logical_or(dones.bool(), truncateds.bool()).view(total, -1)
        resets = resets.to(device=device, dtype=reward_dtype)
        if resets.size(1) != 1:
            resets = resets[:, :1]

    last_actions = td.get("last_actions")
    if last_actions is not None:
        last_actions = last_actions.view(total, -1).to(device=device, dtype=action_dtype)
    else:
        actions = td.get("actions")
        if actions is not None:
            actions = actions.view(batch_size, time_steps, -1).to(device=device, dtype=action_dtype)
            prev = zeros_factory((batch_size, time_steps, actions.size(-1)), device, action_dtype)
            if time_steps > 1:
                prev[:, 1:] = actions[:, :-1]
            last_actions = prev.view(total, -1)
        else:
            last_actions = zeros_factory((total, action_dim), device, action_dtype)

    if last_actions.size(1) != action_dim:
        dim = last_actions.size(1)
        if dim > action_dim:
            last_actions = last_actions[:, :action_dim]
        else:
            pad = zeros_factory((total, action_dim - dim), device, action_dtype)
            last_actions = torch.cat([last_actions, pad], dim=1)

    return reward, resets, last_actions


def _extract_first_available(td: TensorDict, keys: Sequence[str]) -> torch.Tensor | None:
    for key in keys:
        value = td.get(key, None)
        if value is not None:
            return value
    return None


__all__ = ["prepare_auxiliary_signals"]
