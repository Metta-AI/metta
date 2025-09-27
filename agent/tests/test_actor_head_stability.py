#!/usr/bin/env python3
"""Test actor head stability with fp16 to prevent NaN generation."""

import logging

import torch
import torch.nn as nn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_actor_head(hidden_size: int, num_actions: int, device: torch.device) -> nn.Linear:
    actor_head = nn.Linear(hidden_size, num_actions)
    nn.init.normal_(actor_head.weight, std=0.001)
    if actor_head.bias is not None:
        nn.init.zeros_(actor_head.bias)

    with torch.no_grad():
        actor_head.weight.clamp_(-0.01, 0.01)
        if actor_head.bias is not None:
            actor_head.bias.clamp_(-0.01, 0.01)

    return actor_head.to(device=device, dtype=torch.float16)


def _assert_no_nans(tensor: torch.Tensor, *, msg: str) -> None:
    assert not torch.isnan(tensor).any(), msg


def test_actor_head_fp16_stability() -> None:
    """Ensure the FP16 actor head stays numerically stable across representative inputs."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 576  # SmolLM2-135M hidden size
    num_actions = 16
    batch_size = 4

    actor_head = _build_actor_head(hidden_size, num_actions, device)

    normal_input = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
    normal_input = torch.clamp(normal_input, -2.0, 2.0)
    logits_normal = actor_head(normal_input)
    _assert_no_nans(logits_normal, msg="NaNs produced for normal FP16 input")

    ones_input = torch.ones(batch_size, hidden_size, device=device, dtype=torch.float16)
    logits_ones = actor_head(ones_input)
    _assert_no_nans(logits_ones, msg="NaNs produced for all-ones FP16 input")

    edge_input = torch.full((batch_size, hidden_size), 5.0, device=device, dtype=torch.float16)
    logits_edge = actor_head(edge_input)
    _assert_no_nans(logits_edge, msg="NaNs produced for high-magnitude FP16 input")

    weight_min = actor_head.weight.min().item()
    weight_max = actor_head.weight.max().item()
    assert abs(weight_min) <= 0.01 and abs(weight_max) <= 0.01, "Actor head weights exceeded FP16-safe bounds"


if __name__ == "__main__":
    success = test_actor_head_fp16_stability()
    exit(0 if success else 1)
