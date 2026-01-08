#!/usr/bin/env -S uv run python
"""Example showing how to create custom cell types for the cortex architecture."""

import torch
import torch.nn as nn
from cortex import (
    CellConfig,
    CortexStack,
    CortexStackConfig,
    MemoryCell,
    PassThroughBlockConfig,
    register_cell,
)
from cortex.types import MaybeState, ResetMask, Tensor
from pydantic import Field
from tensordict import TensorDict


# Step 1: Define custom cell configuration
class GRUCellConfig(CellConfig):
    """Configuration for a GRU cell."""

    num_layers: int = Field(default=1, ge=1)
    bias: bool = Field(default=True)
    dropout: float = Field(default=0.0, ge=0.0)


# Step 2: Implement and register the custom cell
@register_cell(GRUCellConfig)
class GRUCell(MemoryCell):
    """GRU cell implementation with TensorDict state management."""

    def __init__(self, cfg: GRUCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg
        self.net = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bias=cfg.bias,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,  # Always batch-first
        )
        self.num_layers = cfg.num_layers

    def init_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> TensorDict:
        # Batch-first state: [B, L, H]
        h = torch.zeros(batch, self.num_layers, self.hidden_size, device=device, dtype=dtype)
        return TensorDict({"h": h}, batch_size=[batch])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: ResetMask | None = None,
    ) -> tuple[Tensor, MaybeState]:
        # Handle state
        if state is None:
            batch_size = x.shape[0]  # Always batch-first
            state = self.init_state(batch_size, device=x.device, dtype=x.dtype)

        # Get state and transpose from [B, L, H] to [L, B, H] for nn.GRU
        h = state["h"].transpose(0, 1)  # [B, L, H] -> [L, B, H]

        # Handle input shape - always batch-first
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(1)  # Add seq dimension: [B, H] -> [B, 1, H]

        # Run GRU (always batch-first)
        y, h_new = self.net(x, h)

        # Remove seq dimension if input was not batched
        if not is_batched:
            y = y.squeeze(1)  # [B, 1, H] -> [B, H]

        # Handle resets if provided
        if resets is not None:
            if resets.dim() == 1:  # Batch-level reset
                reset_mask = resets.view(1, -1, 1)  # [B] -> [1, B, 1]
                h_new = torch.where(reset_mask, torch.zeros_like(h_new), h_new)

        # Transpose state back to batch-first: [L, B, H] -> [B, L, H]
        batch_size = x.shape[0]
        h_new_bf = h_new.transpose(0, 1)
        return y, TensorDict({"h": h_new_bf}, batch_size=[batch_size])

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return state

        h = state["h"]  # [B, L, H]
        batch_size = state.batch_size[0] if state.batch_size else h.shape[0]
        reset_mask = mask.view(-1, 1, 1)  # [B] -> [B, 1, 1]
        h_new = torch.where(reset_mask, torch.zeros_like(h), h)
        return TensorDict({"h": h_new}, batch_size=[batch_size])


def test_custom_cell():
    """Test the custom GRU cell."""
    print("Testing Custom Cell Implementation\n" + "=" * 40)

    device = torch.device("cpu")
    dtype = torch.float32
    batch_size = 2
    seq_len = 5
    d_hidden = 64

    # Create a recipe mixing LSTM and GRU cells
    from cortex import LSTMCellConfig

    recipe = CortexStackConfig(
        d_hidden=d_hidden,
        blocks=[
            # LSTM block
            PassThroughBlockConfig(
                cell=LSTMCellConfig(hidden_size=64, num_layers=1),
            ),
            # GRU block (our custom cell)
            PassThroughBlockConfig(
                cell=GRUCellConfig(hidden_size=64, num_layers=2),
            ),
            # Another LSTM block
            PassThroughBlockConfig(
                cell=LSTMCellConfig(hidden_size=64, num_layers=1),
            ),
        ],
        post_norm=True,
    )

    print("Mixed Cell Types Configuration:")
    print(f"  d_hidden: {recipe.d_hidden}")
    print(f"  num_blocks: {len(recipe.blocks)}")
    for i, block in enumerate(recipe.blocks):
        cell_type = type(block.cell).__name__.replace("Config", "")
        print(f"  Block {i}: {cell_type}")
    print()

    # Build the stack - it automatically handles both LSTM and GRU cells!
    stack = CortexStack(recipe)
    print(f"Built stack with {len(stack.blocks)} blocks (mixed cell types)")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_hidden, device=device, dtype=dtype)
    print(f"\nInput shape: {x.shape}")

    state = stack.init_state(batch=batch_size, device=device, dtype=dtype)
    output, new_state = stack(x, state)

    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"

    # Check state structure
    print("\nState structure:")
    for key in new_state.keys():
        block_state = new_state[key]
        if "cell" in block_state:
            cell_state = block_state["cell"]
            print(f"  {key}: {list(cell_state.keys())}")

    print("\n✓ Custom cell test passed!")

    print("\n" + "=" * 40)
    print("How the cell registry system works:")
    print("1. Define your CellConfig subclass with custom parameters")
    print("2. Define your Cell subclass extending MemoryCell")
    print("3. Use @register_cell(YourConfig) on your cell class")
    print("4. Your cell is now usable in any block configuration!")
    print("\nThe system is fully extensible for both blocks AND cells.")


if __name__ == "__main__":
    test_custom_cell()
