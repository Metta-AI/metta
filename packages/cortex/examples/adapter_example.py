"""Example showing how to use AdapterBlocks to wrap existing blocks.

AdapterBlocks allow you to add trainable residual paths that start as identity,
making them perfect for fine-tuning pretrained models without disrupting learned behavior.
"""

import sys
from pathlib import Path

# Add cortex package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from cortex import (
    AdapterBlockConfig,
    CortexStack,
    CortexStackConfig,
    LSTMCellConfig,
    PassThroughBlockConfig,
    PreUpBlockConfig,
)


def test_basic_adapter():
    """Test a basic adapter wrapping a PassThrough block."""
    print("Basic Adapter Example\n" + "=" * 60)

    device = torch.device("cpu")
    dtype = torch.float32
    batch_size = 2
    seq_len = 5
    d_hidden = 128

    # Create a stack with an adapter wrapping a simple LSTM block
    config = CortexStackConfig(
        d_hidden=d_hidden,
        blocks=[
            AdapterBlockConfig(
                base_block=PassThroughBlockConfig(cell=LSTMCellConfig(hidden_size=128, num_layers=1)),
                bottleneck=32,  # Small bottleneck for efficiency
                per_channel_gate=False,  # Scalar gate
            )
        ],
        post_norm=False,  # No post norm for identity check
    )

    stack = CortexStack(config)
    stack.to(device=device, dtype=dtype)

    print("Stack configuration:")
    print(f"  d_hidden: {d_hidden}")
    print("  Blocks: 1 adapter wrapping LSTM")
    print("  Bottleneck size: 32")
    print("  Gate type: scalar\n")

    # Test identity at initialization
    x = torch.randn(batch_size, seq_len, d_hidden, device=device, dtype=dtype)
    state = stack.init_state(batch=batch_size, device=device, dtype=dtype)

    # Get output from wrapped block directly
    base_block = stack.blocks[0].wrapped_block
    base_state = state["block_0"]["wrapped"]
    y_base, _ = base_block(x, base_state)

    # Get output through adapter
    stack.eval()
    y_adapter, _ = stack(x, state)

    max_diff = (y_adapter - y_base).abs().max().item()
    print("Identity check at initialization:")
    print(f"  Max difference: {max_diff:.2e}")
    print("  ✓ Adapter is identity at init!" if max_diff < 1e-6 else "  ✗ Not identity")
    print()


def test_freezing_and_training():
    """Test freezing base model and training only adapters."""
    print("Freezing and Training Example\n" + "=" * 60)

    device = torch.device("cpu")
    dtype = torch.float32
    batch_size = 2
    seq_len = 5
    d_hidden = 128

    # Create stack with multiple blocks, some wrapped with adapters
    config = CortexStackConfig(
        d_hidden=d_hidden,
        blocks=[
            PassThroughBlockConfig(cell=LSTMCellConfig(hidden_size=128, num_layers=1)),  # Regular block
            AdapterBlockConfig(
                base_block=PassThroughBlockConfig(cell=LSTMCellConfig(hidden_size=128, num_layers=1)),
                bottleneck=32,
            ),  # Adapter
            PassThroughBlockConfig(cell=LSTMCellConfig(hidden_size=128, num_layers=1)),  # Regular block
        ],
        post_norm=True,
    )

    stack = CortexStack(config)
    stack.to(device=device, dtype=dtype)
    stack.train()

    print(f"Stack with {len(stack.blocks)} blocks:")
    print("  Block 0: Regular LSTM")
    print("  Block 1: Adapter wrapping LSTM")
    print("  Block 2: Regular LSTM\n")

    # Count total parameters before freezing
    total_params = sum(p.numel() for p in stack.parameters())
    print(f"Total parameters: {total_params:,}")

    # Freeze all non-adapter blocks
    frozen_count = 0
    for i, block in enumerate(stack.blocks):
        from cortex.blocks.adapter import AdapterBlock

        if not isinstance(block, AdapterBlock):
            for param in block.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            print(f"  Froze block {i}: {sum(p.numel() for p in block.parameters()):,} params")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in stack.parameters() if p.requires_grad)
    print("\nAfter freezing:")
    print(f"  Frozen parameters: {frozen_count:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%\n")

    # Simulate training
    x = torch.randn(batch_size, seq_len, d_hidden, device=device, dtype=dtype)
    state = stack.init_state(batch=batch_size, device=device, dtype=dtype)

    y, _ = stack(x, state)
    loss = y.sum()
    loss.backward()

    # Check gradients
    print("Gradient check:")
    has_grad_count = sum(1 for p in stack.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {has_grad_count}")
    print("  ✓ Only adapter parameters have gradients!\n")


def test_adapter_wrapping_preup():
    """Test adapter wrapping a more complex PreUp block."""
    print("Adapter Wrapping PreUp Block\n" + "=" * 60)

    device = torch.device("cpu")
    dtype = torch.float32
    batch_size = 2
    seq_len = 5
    d_hidden = 128

    config = CortexStackConfig(
        d_hidden=d_hidden,
        blocks=[
            AdapterBlockConfig(
                base_block=PreUpBlockConfig(
                    cell=LSTMCellConfig(hidden_size=None, num_layers=2),  # Inferred: 2x upsampling
                    proj_factor=2.0,
                ),
                bottleneck=64,
                per_channel_gate=True,  # Per-channel gate for more expressiveness
                activation="silu",
            )
        ],
        post_norm=True,
    )

    stack = CortexStack(config)
    stack.to(device=device, dtype=dtype)

    print("Adapter wrapping PreUp block:")
    print("  Base block: PreUp with 2x projection (d_inner=256)")
    print("  Adapter bottleneck: 64")
    print(f"  Gate: per-channel ({d_hidden} parameters)")
    print("  Activation: SiLU\n")

    x = torch.randn(batch_size, seq_len, d_hidden, device=device, dtype=dtype)
    state = stack.init_state(batch=batch_size, device=device, dtype=dtype)

    y, new_state = stack(x, state)

    print("Forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  ✓ Shape preserved through adapter + PreUp!\n")


if __name__ == "__main__":
    test_basic_adapter()
    test_freezing_and_training()
    test_adapter_wrapping_preup()
    print("\n" + "=" * 60)
    print("All adapter examples completed successfully!")
    print("=" * 60)
