#!/usr/bin/env python3
"""Minimal script to test DDP unused parameters issue.

This script tests whether DDP training works correctly with various configurations.
Run with torchrun to simulate multi-GPU training:

    # Test with 2 processes (works on CPU too)
    uv run torchrun --nproc-per-node=2 --master-port=29501 scripts/test_ddp_unused_params.py

The script will:
1. Create a simple policy and wrap it in DDP
2. Run a forward/backward pass where some outputs are unused
3. This simulates what happens in ppo_actor.py and sliced_scripted_cloner.py

Expected behavior:
- With find_unused_parameters=False (current): hangs or errors without the 0*sum hack
- With find_unused_parameters=True: works without the hack
- With static_graph=True: works if graph is consistent
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimplePolicy(nn.Module):
    """A simple policy that outputs multiple tensors, not all used in every loss."""

    def __init__(self, input_dim=64, hidden_dim=128, num_actions=10):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_actions)  # Used by actor loss
        self.critic = nn.Linear(hidden_dim, 1)  # Used by critic loss
        self.aux_head = nn.Linear(hidden_dim, 32)  # Sometimes unused

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        logits = self.actor(hidden)
        values = self.critic(hidden)
        aux = self.aux_head(hidden)
        return {"logits": logits, "values": values, "aux": aux, "hidden": hidden}


def test_ddp_unused_params(
    find_unused_parameters: bool = False,
    static_graph: bool = False,
    use_hack: bool = True,
):
    """Test DDP with unused parameters.

    Args:
        find_unused_parameters: DDP setting to auto-detect unused params
        static_graph: DDP setting for static computation graphs
        use_hack: Whether to use the 0*sum hack to include unused params
    """
    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")  # gloo works on CPU

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing DDP with:")
        print(f"  find_unused_parameters={find_unused_parameters}")
        print(f"  static_graph={static_graph}")
        print(f"  use_hack={use_hack}")
        print(f"  world_size={world_size}")
        print(f"{'='*60}\n")

    # Create model and wrap in DDP
    model = SimplePolicy()

    ddp_kwargs = {
        "find_unused_parameters": find_unused_parameters,
    }
    if static_graph:
        ddp_kwargs["static_graph"] = True

    model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate training loop
    num_iterations = 5
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Create dummy input (different on each rank to simulate real training)
        x = torch.randn(32, 64) + rank * 0.1

        # Forward pass
        outputs = model(x)

        # Compute loss using ONLY logits (not values or aux)
        # This simulates ppo_actor which uses act_log_prob but not values
        loss = outputs["logits"].mean()

        # The hack: include unused outputs with 0 weight
        if use_hack:
            for key, value in outputs.items():
                if key != "logits" and isinstance(value, torch.Tensor):
                    if value.requires_grad:
                        loss = loss + 0.0 * value.sum()

        # Backward pass - this is where DDP hangs without proper config
        loss.backward()

        # Optimizer step
        optimizer.step()

        if rank == 0:
            print(f"  Iteration {i+1}/{num_iterations}: loss={loss.item():.4f}")

    if rank == 0:
        print(f"\n✅ SUCCESS: Training completed without hanging!\n")

    dist.barrier()
    return True


def main():
    """Run tests with different configurations."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--find-unused", action="store_true", help="Use find_unused_parameters=True")
    parser.add_argument("--static-graph", action="store_true", help="Use static_graph=True")
    parser.add_argument("--no-hack", action="store_true", help="Disable the 0*sum hack")
    args = parser.parse_args()

    try:
        test_ddp_unused_params(
            find_unused_parameters=args.find_unused,
            static_graph=args.static_graph,
            use_hack=not args.no_hack,
        )
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

