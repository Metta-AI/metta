"""Test LayerNorm and skip connections in PreUp and PostUp blocks."""

import torch
import torch.nn as nn
from cortex.blocks.postup import PostUpBlock
from cortex.blocks.preup import PreUpBlock
from cortex.cells.lstm import LSTMCell
from cortex.config import LSTMCellConfig, PostUpBlockConfig, PreUpBlockConfig


def test_preup_block_has_layernorm():
    """Verify PreUpBlock has LayerNorm."""
    config = PreUpBlockConfig(cell=LSTMCellConfig(hidden_size=256), proj_factor=2.0)
    d_hidden = 128
    cell = LSTMCell(LSTMCellConfig(hidden_size=256))

    block = PreUpBlock(config, d_hidden, cell)

    # Check LayerNorm exists
    assert hasattr(block, "norm")
    assert isinstance(block.norm, nn.LayerNorm)
    assert block.norm.normalized_shape[0] == d_hidden
    assert block.norm.elementwise_affine == True
    assert block.norm.bias is None


def test_postup_block_has_layernorms():
    """Verify PostUpBlock has both LayerNorms."""
    config = PostUpBlockConfig(cell=LSTMCellConfig(hidden_size=128), proj_factor=2.0)
    d_hidden = 128
    cell = LSTMCell(LSTMCellConfig(hidden_size=128))

    block = PostUpBlock(config, d_hidden, cell)

    # Check both LayerNorms exist
    assert hasattr(block, "norm")
    assert isinstance(block.norm, nn.LayerNorm)
    assert block.norm.normalized_shape[0] == d_hidden

    assert hasattr(block, "ffn_norm")
    assert isinstance(block.ffn_norm, nn.LayerNorm)
    assert block.ffn_norm.normalized_shape[0] == d_hidden


def test_preup_residual_connection():
    """Verify PreUpBlock properly applies residual connection."""
    config = PreUpBlockConfig(cell=LSTMCellConfig(hidden_size=256), proj_factor=2.0)
    d_hidden = 128
    cell = LSTMCell(LSTMCellConfig(hidden_size=256))

    block = PreUpBlock(config, d_hidden, cell)

    # Create input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_hidden)

    # Initialize state
    state = block.init_state(batch=batch_size, device="cpu", dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        # Set output projection to zero to test residual
        block.out_proj.weight.zero_()
        block.out_proj.bias.zero_() if block.out_proj.bias is not None else None

        y, _ = block(x, state)

        # Output should be close to input due to residual connection
        # (the processed part contributes zero due to zeroed projection)
        assert torch.allclose(y, x, atol=1e-5)


def test_postup_residual_connections():
    """Verify PostUpBlock properly applies dual residual connections."""
    config = PostUpBlockConfig(cell=LSTMCellConfig(hidden_size=128), proj_factor=2.0)
    d_hidden = 128
    cell = LSTMCell(LSTMCellConfig(hidden_size=128))

    block = PostUpBlock(config, d_hidden, cell)

    # Create input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_hidden)

    # Initialize state
    state = block.init_state(batch=batch_size, device="cpu", dtype=torch.float32)

    # Forward pass - test that residuals are preserved
    with torch.no_grad():
        y, _ = block(x, state)

        # Output should have similar magnitude to input due to residual connections
        assert y.shape == x.shape
        # Check that norms are in reasonable range (not exploding/vanishing)
        input_norm = torch.norm(x, dim=-1).mean()
        output_norm = torch.norm(y, dim=-1).mean()
        ratio = output_norm / input_norm
        assert 0.5 < ratio < 3.0, f"Output/input norm ratio {ratio} is out of expected range"


if __name__ == "__main__":
    test_preup_block_has_layernorm()
    test_postup_block_has_layernorms()
    test_preup_residual_connection()
    test_postup_residual_connections()
    print("All tests passed!")
