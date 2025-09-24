#!/usr/bin/env python3
"""Test multinomial sampling with fp16 to catch device-side assert issues."""

import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multinomial_fp16():
    """Test that multinomial works correctly with fp16 tensors."""
    print("Testing multinomial with fp16...")

    # Create test scenarios that might trigger device-side asserts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    batch_size = 4
    num_actions = 10

    # Test case 1: Normal logits converted to fp16
    print("\n1. Testing normal fp16 logits...")
    logits_fp32 = torch.randn(batch_size, num_actions, device=device, dtype=torch.float32)
    logits_fp16 = logits_fp32.to(torch.float16)

    # Apply the same processing as our fixed forward_inference
    logits_clamped = torch.clamp(logits_fp16, min=-10.0, max=10.0)
    log_probs = torch.nn.functional.log_softmax(logits_clamped, dim=-1)
    action_probs = torch.exp(log_probs)

    # Use fp16-safe minimum
    min_prob = 1e-6 if action_probs.dtype == torch.float16 else 1e-8
    action_probs = torch.clamp(action_probs, min=min_prob, max=1.0)
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

    try:
        actions = torch.multinomial(action_probs, num_samples=1).view(-1)
        print(f"✅ Normal fp16 multinomial succeeded: {actions}")
    except Exception as e:
        print(f"❌ Normal fp16 multinomial failed: {e}")
        return False

    # Test case 2: Extreme logits that might cause issues
    print("\n2. Testing extreme fp16 logits...")
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

    # Apply same processing
    extreme_clamped = torch.clamp(extreme_logits, min=-10.0, max=10.0)
    extreme_log_probs = torch.nn.functional.log_softmax(extreme_clamped, dim=-1)
    extreme_action_probs = torch.exp(extreme_log_probs)
    extreme_action_probs = torch.clamp(extreme_action_probs, min=1e-6, max=1.0)
    extreme_action_probs = extreme_action_probs / extreme_action_probs.sum(dim=-1, keepdim=True)

    try:
        extreme_actions = torch.multinomial(extreme_action_probs, num_samples=1).view(-1)
        print(f"✅ Extreme fp16 multinomial succeeded: {extreme_actions}")
    except Exception as e:
        print(f"❌ Extreme fp16 multinomial failed: {e}")
        return False

    # Test case 3: Very small probabilities (edge of fp16 precision)
    print("\n3. Testing edge-case fp16 probabilities...")
    tiny_logits = torch.full((batch_size, num_actions), -8.0, device=device, dtype=torch.float16)
    # Make one action slightly more likely
    tiny_logits[:, 0] = -7.9

    tiny_log_probs = torch.nn.functional.log_softmax(tiny_logits, dim=-1)
    tiny_action_probs = torch.exp(tiny_log_probs)
    tiny_action_probs = torch.clamp(tiny_action_probs, min=1e-6, max=1.0)
    tiny_action_probs = tiny_action_probs / tiny_action_probs.sum(dim=-1, keepdim=True)

    print(f"Tiny probabilities range: {tiny_action_probs.min().item():.8f} to {tiny_action_probs.max().item():.8f}")

    try:
        tiny_actions = torch.multinomial(tiny_action_probs, num_samples=1).view(-1)
        print(f"✅ Tiny fp16 probabilities multinomial succeeded: {tiny_actions}")
    except Exception as e:
        print(f"❌ Tiny fp16 probabilities multinomial failed: {e}")
        return False

    print("\n✅ All multinomial fp16 tests passed!")
    return True


if __name__ == "__main__":
    success = test_multinomial_fp16()
    exit(0 if success else 1)
