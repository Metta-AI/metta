"""Tests for LSTM cell implementation."""

import os
import sys
from pathlib import Path

import torch

# Make cortex package importable relative to this test
PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(PKG_ROOT / "src"))

from cortex.cells.lstm import LSTMCell  # noqa: E402
from cortex.config import LSTMCellConfig  # noqa: E402


def get_test_device():
    """Get the appropriate device for testing (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def test_lstm_sequence_forward():
    """Test LSTM cell with sequence input."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    # Test sequence processing
    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    # Check shapes
    assert y.shape == (B, T, H), f"Expected output shape {(B, T, H)}, got {y.shape}"
    assert state is not None, "State should not be None"
    assert "h" in state, "State should contain 'h'"
    assert "c" in state, "State should contain 'c'"
    assert state["h"].shape == (B, 1, H), f"Expected h shape {(B, 1, H)}, got {state['h'].shape}"
    assert state["c"].shape == (B, 1, H), f"Expected c shape {(B, 1, H)}, got {state['c'].shape}"


def test_lstm_single_step():
    """Test LSTM cell with single-step input."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    # Test single-step processing
    x = torch.randn(B, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    # Check shapes
    assert y.shape == (B, H), f"Expected output shape {(B, H)}, got {y.shape}"
    assert state is not None, "State should not be None"
    assert "h" in state, "State should contain 'h'"
    assert "c" in state, "State should contain 'c'"


def test_lstm_sequential_vs_parallel():
    """Test that sequential and parallel processing are consistent."""
    torch.manual_seed(0)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 16  # sequence length
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel: process entire sequence at once
    with torch.no_grad():
        y_parallel, _ = cell(x, state=None)

    # Sequential: process one timestep at a time
    state = None
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = cell(x[:, t, :], state)
            y_steps.append(y_t)
    y_sequential = torch.stack(y_steps, dim=1)

    # Check shapes match
    assert y_parallel.shape == y_sequential.shape

    # Check outputs are close (should be identical for LSTM)
    torch.testing.assert_close(y_parallel, y_sequential, rtol=1e-5, atol=1e-5)


def test_lstm_multi_layer():
    """Test LSTM with multiple layers."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 64  # hidden size
    num_layers = 3

    cfg = LSTMCellConfig(hidden_size=H, num_layers=num_layers, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    # Check shapes
    assert y.shape == (B, T, H), f"Expected output shape {(B, T, H)}, got {y.shape}"
    assert state["h"].shape == (B, num_layers, H), f"Expected h shape {(B, num_layers, H)}, got {state['h'].shape}"
    assert state["c"].shape == (B, num_layers, H), f"Expected c shape {(B, num_layers, H)}, got {state['c'].shape}"


def test_lstm_with_projection():
    """Test LSTM with projection layer."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 64  # hidden size
    proj_size = 32

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, proj_size=proj_size, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    # Output should have proj_size instead of H
    assert y.shape == (B, T, proj_size), f"Expected output shape {(B, T, proj_size)}, got {y.shape}"
    assert state["h"].shape == (B, 1, proj_size), f"Expected h shape {(B, 1, proj_size)}, got {state['h'].shape}"
    assert state["c"].shape == (B, 1, H), f"Expected c shape {(B, 1, H)}, got {state['c'].shape}"


def test_lstm_state_reset():
    """Test LSTM state reset functionality."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 4  # batch size
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)

    # Create initial state
    state = cell.init_state(batch=B, device=device, dtype=dtype)

    # Set some values in state
    state["h"].fill_(1.0)
    state["c"].fill_(2.0)

    # Create reset mask - reset first 2 samples
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    mask[:2] = True

    # Reset state
    new_state = cell.reset_state(state, mask)

    # Check that new_state is not None and has correct keys
    assert new_state is not None
    assert "h" in new_state
    assert "c" in new_state

    # Check that first 2 samples are reset to zero
    new_h = new_state["h"]
    new_c = new_state["c"]
    assert new_h is not None and new_c is not None
    assert torch.allclose(new_h[:2], torch.zeros_like(new_h[:2]))
    assert torch.allclose(new_c[:2], torch.zeros_like(new_c[:2]))

    # Check that last 2 samples are unchanged
    assert torch.allclose(new_h[2:], state["h"][2:])
    assert torch.allclose(new_c[2:], state["c"][2:])


def test_lstm_with_resets():
    """Test LSTM with per-timestep resets."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Create reset mask - reset at timestep 5 for first sample
    resets = torch.zeros(B, T, dtype=torch.bool, device=device)
    resets[0, 5] = True

    with torch.no_grad():
        y, state = cell(x, state=None, resets=resets)

    # Check shapes
    assert y.shape == (B, T, H), f"Expected output shape {(B, T, H)}, got {y.shape}"
    assert state is not None, "State should not be None"


def test_lstm_gradient_flow():
    """Test that gradients flow through LSTM."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 64  # hidden size

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.train()

    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    y, _ = cell(x, state=None)

    # Compute loss and backward
    loss = y.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert any(p.grad is not None for p in cell.parameters()), "Cell parameters should have gradients"


if __name__ == "__main__":
    # Run tests
    test_lstm_sequence_forward()
    test_lstm_single_step()
    test_lstm_sequential_vs_parallel()
    test_lstm_multi_layer()
    test_lstm_with_projection()
    test_lstm_state_reset()
    test_lstm_with_resets()
    test_lstm_gradient_flow()
    print("All LSTM tests passed!")
