"""Tests for LSTM cell implementation."""

import pytest
import torch
from cortex.cells.lstm import LSTMCell
from cortex.config import LSTMCellConfig
from cortex.kernels.pytorch.lstm import lstm_sequence_pytorch
from cortex.utils import TRITON_AVAILABLE


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
    H = 32  # hidden size

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
    H = 32  # hidden size

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
    H = 32  # hidden size

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


def test_lstm_multi_layer_unsupported():
    """Multi-layer configuration should raise now that kernels are single-layer only."""

    cfg = LSTMCellConfig(hidden_size=64, num_layers=2, dropout=0.0)
    with pytest.raises(ValueError):
        LSTMCell(cfg)


def test_lstm_projection_unsupported():
    """Projection size >0 is not supported by the fused implementation."""

    cfg = LSTMCellConfig(hidden_size=64, num_layers=1, proj_size=16, dropout=0.0)
    with pytest.raises(ValueError):
        LSTMCell(cfg)


def test_lstm_state_reset():
    """Test LSTM state reset functionality."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 4  # batch size
    H = 32  # hidden size

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

    # Remaining samples should be unchanged
    assert torch.allclose(new_h[2:], state["h"][2:])
    assert torch.allclose(new_c[2:], state["c"][2:])


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="Triton backend unavailable")
def test_lstm_reset_forward_backward_match_backends():
    """Ensure Triton reset behaviour matches PyTorch forward/backward on CUDA."""
    from cortex.kernels.triton.lstm import lstm_sequence_triton

    torch.manual_seed(1234)

    device = torch.device("cuda")
    dtype = torch.float32
    B, T, H = 3, 5, 32

    cfg = LSTMCellConfig(hidden_size=H, num_layers=1, dropout=0.0)
    cell = LSTMCell(cfg).to(device=device, dtype=dtype)
    cell.net.reset_parameters()
    net = cell.net

    state0 = cell.init_state(batch=B, device=device, dtype=dtype)
    h0 = state0.get("h")
    c0 = state0.get("c")
    assert h0 is not None and c0 is not None

    resets = torch.zeros(B, T, dtype=torch.bool, device=device)
    resets[0, 2] = True
    resets[1, 0] = True

    x_base = torch.randn(B, T, H, device=device, dtype=dtype)

    # PyTorch reference
    net.zero_grad(set_to_none=True)
    x_pt = x_base.clone().detach().requires_grad_(True)
    y_pt, hn_pt, cn_pt = lstm_sequence_pytorch(lstm=net, x_seq=x_pt, h0_bf=h0.clone(), c0_bf=c0.clone(), resets=resets)
    loss_pt = y_pt.square().sum() + hn_pt.square().sum() + cn_pt.square().sum()
    loss_pt.backward()
    grad_x_pt = x_pt.grad.detach().clone()
    grad_wih_pt = net.weight_ih_l0.grad.detach().clone()
    grad_whh_pt = net.weight_hh_l0.grad.detach().clone()
    grad_bias_pt = (
        (net.bias_ih_l0.grad + net.bias_hh_l0.grad).detach().clone()
        if net.bias
        else torch.zeros_like(grad_wih_pt[:, 0])
    )
    net.zero_grad(set_to_none=True)

    # Triton implementation
    x_tr = x_base.clone().detach().requires_grad_(True)
    y_tr, hn_tr, cn_tr = lstm_sequence_triton(lstm=net, x_seq=x_tr, h0_bf=h0.clone(), c0_bf=c0.clone(), resets=resets)
    loss_tr = y_tr.square().sum() + hn_tr.square().sum() + cn_tr.square().sum()
    loss_tr.backward()
    grad_x_tr = x_tr.grad.detach().clone()
    grad_wih_tr = net.weight_ih_l0.grad.detach().clone()
    grad_whh_tr = net.weight_hh_l0.grad.detach().clone()
    grad_bias_tr = (
        (net.bias_ih_l0.grad + net.bias_hh_l0.grad).detach().clone()
        if net.bias
        else torch.zeros_like(grad_wih_tr[:, 0])
    )

    tol_values = {"rtol": 1e-3, "atol": 1e-2}
    tol_grads = {"rtol": 1e-3, "atol": 1e-1}

    torch.testing.assert_close(y_pt, y_tr, **tol_values)
    torch.testing.assert_close(hn_pt, hn_tr, **tol_values)
    torch.testing.assert_close(cn_pt, cn_tr, **tol_values)

    torch.testing.assert_close(grad_x_pt, grad_x_tr, **tol_grads)
    torch.testing.assert_close(grad_wih_pt, grad_wih_tr, **tol_grads)
    torch.testing.assert_close(grad_whh_pt, grad_whh_tr, **tol_grads)
    torch.testing.assert_close(grad_bias_pt, grad_bias_tr, **tol_grads)


def test_lstm_with_resets():
    """Test LSTM with per-timestep resets."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 10  # sequence length
    H = 32  # hidden size

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
    # Run all tests
    test_lstm_sequence_forward()
    test_lstm_single_step()
    test_lstm_sequential_vs_parallel()
    test_lstm_multi_layer_unsupported()
    test_lstm_projection_unsupported()
    test_lstm_state_reset()
    if torch.cuda.is_available() and TRITON_AVAILABLE:
        test_lstm_reset_forward_backward_match_backends()
    test_lstm_with_resets()
    test_lstm_gradient_flow()
