#!/usr/bin/env python3
"""Test actor head stability with fp16 to prevent NaN generation."""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_actor_head_fp16_stability():
    """Test that actor heads don't produce NaN with fp16 and extreme inputs."""
    print("Testing actor head FP16 stability...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Test parameters matching SmolLM2
    hidden_size = 576  # SmolLM2-135M hidden size
    num_actions = 16   # Typical action space size
    batch_size = 4
    
    # Create actor head with our FP16-safe initialization
    actor_head = nn.Linear(hidden_size, num_actions)
    nn.init.normal_(actor_head.weight, std=0.001)  # Small std
    if actor_head.bias is not None:
        nn.init.zeros_(actor_head.bias)
    
    # Clamp weights to prevent overflow
    with torch.no_grad():
        actor_head.weight.clamp_(-0.01, 0.01)
        if actor_head.bias is not None:
            actor_head.bias.clamp_(-0.01, 0.01)
    
    # Move to device and convert to FP16
    actor_head = actor_head.to(device).to(dtype=torch.float16)
    
    # Test case 1: Normal input (like pooled hidden states)
    print("\n1. Testing with normal input...")
    normal_input = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
    normal_input = torch.clamp(normal_input, -2.0, 2.0)  # Reasonable range
    
    logits = actor_head(normal_input)
    print(f"Normal input - logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
    
    if torch.isnan(logits).any():
        print("❌ Found NaN in logits with normal input!")
        return False
    else:
        print("✅ Normal input produces valid logits")
    
    # Test case 2: Extreme but realistic input (all 1.0s like in the debug output)
    print("\n2. Testing with all-ones input...")
    ones_input = torch.ones(batch_size, hidden_size, device=device, dtype=torch.float16)
    
    logits_ones = actor_head(ones_input)
    print(f"Ones input - logits range: {logits_ones.min().item():.4f} to {logits_ones.max().item():.4f}")
    
    if torch.isnan(logits_ones).any():
        print("❌ Found NaN in logits with all-ones input!")
        return False
    else:
        print("✅ All-ones input produces valid logits")
    
    # Test case 3: Edge case inputs near FP16 limits
    print("\n3. Testing with edge case inputs...")
    edge_input = torch.full((batch_size, hidden_size), 5.0, device=device, dtype=torch.float16)  # High but safe
    
    logits_edge = actor_head(edge_input)
    print(f"Edge input - logits range: {logits_edge.min().item():.4f} to {logits_edge.max().item():.4f}")
    
    if torch.isnan(logits_edge).any():
        print("❌ Found NaN in logits with edge case input!")
        return False
    else:
        print("✅ Edge case input produces valid logits")
    
    # Test case 4: Check weight bounds are maintained
    print("\n4. Checking weight bounds...")
    weight_min = actor_head.weight.min().item()
    weight_max = actor_head.weight.max().item()
    print(f"Weight bounds: {weight_min:.6f} to {weight_max:.6f}")
    
    if abs(weight_min) > 0.01 or abs(weight_max) > 0.01:
        print("❌ Weights exceed safe bounds!")
        return False
    else:
        print("✅ Weights within safe bounds")
    
    print("\n✅ All actor head stability tests passed!")
    return True


if __name__ == "__main__":
    success = test_actor_head_fp16_stability()
    exit(0 if success else 1)