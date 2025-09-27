#!/usr/bin/env python3
"""Test multinomial sampling with fp16 to catch device-side assert issues."""

import logging

import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clamped_probs(logits: torch.Tensor) -> torch.Tensor:
    logits_clamped = torch.clamp(logits, min=-10.0, max=10.0)
    log_probs = torch.nn.functional.log_softmax(logits_clamped, dim=-1)
    action_probs = torch.exp(log_probs)
    min_prob = 1e-6 if action_probs.dtype == torch.float16 else 1e-8
    action_probs = torch.clamp(action_probs, min=min_prob, max=1.0)
    return action_probs / action_probs.sum(dim=-1, keepdim=True)


def _sample_and_assert_valid(probs: torch.Tensor) -> torch.Tensor:
    actions = torch.multinomial(probs, num_samples=1).view(-1)
    assert actions.shape[0] == probs.shape[0]
    assert (actions >= 0).all() and (actions < probs.shape[1]).all()
    return actions


def test_multinomial_fp16() -> None:
    """Verify multinomial sampling behaves with FP16 logits in representative regimes."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_actions = 10

    logits_fp32 = torch.randn(batch_size, num_actions, device=device, dtype=torch.float32)
    logits_fp16 = logits_fp32.to(torch.float16)
    probs_normal = _clamped_probs(logits_fp16)
    _sample_and_assert_valid(probs_normal)

    extreme_logits = torch.tensor(
        [
            [10.0, -10.0, 0.0, 5.0, -5.0, 15.0, -15.0, 2.0, -2.0, 0.5],
            [-20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [100.0, -100.0, 50.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float16,
    )
    probs_extreme = _clamped_probs(extreme_logits)
    _sample_and_assert_valid(probs_extreme)

    tiny_logits = torch.full((batch_size, num_actions), -8.0, device=device, dtype=torch.float16)
    tiny_logits[:, 0] = -7.9
    probs_tiny = _clamped_probs(tiny_logits)
    _sample_and_assert_valid(probs_tiny)


if __name__ == "__main__":
    success = test_multinomial_fp16()
    exit(0 if success else 1)
