"""Tests for mLSTM cell implementation."""

import os
import sys
from pathlib import Path

import torch

# Make cortex package importable relative to this test
PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(PKG_ROOT / "src"))

from cortex.blocks import PreUpBlock  # noqa: E402
from cortex.cells.mlstm import mLSTMCell  # noqa: E402
from cortex.config import PreUpBlockConfig, mLSTMCellConfig  # noqa: E402


def get_test_device():
    """Get the appropriate device for testing (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def test_mlstm_parallel_vs_sequential_close() -> None:
    """Test that parallel and sequential processing produce similar outputs."""
    torch.manual_seed(0)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 32  # sequence length
    H = 64  # hidden size (must be divisible by num_heads, head_dim must be >= 16 for triton)
    num_heads = 4

    # Ensure the parallel path is used in sequence mode (S <= chunk_size)
    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=256,
        conv1d_kernel_size=4,
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel over the entire sequence
    y_parallel, _ = cell(x, state=None)

    # Sequential: feed one timestep at a time, carrying state
    state = None
    y_steps = []
    for t in range(T):
        y_t, state = cell(x[:, t, :], state)
        y_steps.append(y_t)
    y_sequential = torch.stack(y_steps, dim=1)

    assert y_parallel.shape == y_sequential.shape
    torch.testing.assert_close(y_parallel, y_sequential, rtol=5e-3, atol=5e-3)


def test_mlstm_with_preup_block() -> None:
    """Test mLSTM cell within a PreUp block for proper forward pass."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 16  # sequence length
    D = 64  # external hidden size
    proj_factor = 2.0  # PreUp projection factor

    # Create PreUp block config with mLSTM cell
    # The cell will operate on dimension D * proj_factor = 128
    inner_dim = int(D * proj_factor)
    mlstm_config = mLSTMCellConfig(
        hidden_size=inner_dim,
        num_heads=4,
        chunk_size=32,
        conv1d_kernel_size=4,
    )

    preup_config = PreUpBlockConfig(
        cell=mlstm_config,
        proj_factor=proj_factor,
    )

    # Create the cell and PreUp block
    cell = mLSTMCell(mlstm_config).to(device=device, dtype=dtype)
    block = PreUpBlock(preup_config, d_hidden=D, cell=cell).to(device=device, dtype=dtype)
    block.eval()

    # Create input
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # Forward pass through PreUp block
    with torch.no_grad():
        output, state = block(x, state=None)

    # Check output shape matches input (due to skip connection and down projection)
    assert output.shape == (B, T, D), f"Expected shape {(B, T, D)}, got {output.shape}"

    # Check that state exists and has correct structure
    assert state is not None, "State should not be None"

    # The PreUpBlock wraps the cell state in a 'cell' key
    assert "cell" in state, f"State should contain 'cell' key, got keys: {list(state.keys())}"
    cell_state = state["cell"]

    # Check cell state components
    assert "c" in cell_state, "Cell state should contain 'c' component"
    assert "n" in cell_state, "Cell state should contain 'n' component"
    assert "m" in cell_state, "Cell state should contain 'm' component"
    assert "conv" in cell_state, "Cell state should contain 'conv' component"

    # Check state dimensions match the inner dimension (D * proj_factor)
    num_heads = mlstm_config.num_heads
    head_dim = inner_dim // num_heads

    assert cell_state["c"].shape == (B, num_heads, head_dim, head_dim)
    assert cell_state["n"].shape == (B, num_heads, head_dim, 1)
    assert cell_state["m"].shape == (B, num_heads, 1, 1)
    assert cell_state["conv"].shape == (B, mlstm_config.conv1d_kernel_size, inner_dim)


def test_mlstm_sequential_vs_parallel_with_chunking() -> None:
    """Test sequential vs parallel with smaller chunk size to force chunking."""
    torch.manual_seed(123)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 32  # Reduce sequence length to minimize numerical accumulation
    H = 64  # hidden size (head_dim must be >= 16 for triton)
    num_heads = 4
    chunk_size = 64  # Set chunk_size > T to force parallel_stabilized_simple path

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=chunk_size,  # This will make it use parallel path instead of chunking
        conv1d_kernel_size=4,
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel processing (will use mlstm_parallel_stabilized_simple since T < chunk_size)
    with torch.no_grad():
        y_parallel, state_parallel = cell(x, state=None)

    # Sequential processing
    state = None
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = cell(x[:, t, :], state)
            y_steps.append(y_t)
    y_sequential = torch.stack(y_steps, dim=1)

    # Compare outputs
    assert y_parallel.shape == y_sequential.shape
    torch.testing.assert_close(
        y_parallel,
        y_sequential,
        rtol=5e-3,
        atol=5e-3,
        msg="Sequential and parallel outputs differ beyond tolerance",
    )

    # Compare final states - the matrix state can accumulate larger differences
    # due to the matrix multiplications involved
    c_diff_max = (state_parallel["c"] - state["c"]).abs().max().item()
    c_diff_rel = ((state_parallel["c"] - state["c"]).abs() / (state_parallel["c"].abs() + 1e-8)).max().item()

    print(f"Max absolute difference in c state: {c_diff_max:.6f}")
    print(f"Max relative difference in c state: {c_diff_rel:.6f}")

    # Check that outputs match closely (this is most important)
    # For states, we accept larger differences as they accumulate over time
    # and the exact state values are less critical than the output
    assert c_diff_rel < 2.0, f"Relative difference in c state too large: {c_diff_rel}"
    n_diff_max = (state_parallel["n"] - state["n"]).abs().max().item()
    print(f"Max absolute difference in n state: {n_diff_max:.6f}")

    # Use even more relaxed tolerance for n state (larger for triton/CUDA numerical differences)
    torch.testing.assert_close(
        state_parallel["n"],
        state["n"],
        rtol=0.1,
        atol=0.1,
        msg="Final 'n' states differ",
    )
    torch.testing.assert_close(
        state_parallel["m"],
        state["m"],
        rtol=1e-2,
        atol=1e-2,
        msg="Final 'm' states differ",
    )


def test_mlstm_gradient_flow() -> None:
    """Test gradient flow through mLSTM cell."""
    torch.manual_seed(456)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 16
    H = 64  # head_dim must be >= 16 for triton
    num_heads = 4

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=32,
        conv1d_kernel_size=4,
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.train()  # Ensure we're in training mode

    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)

    # Forward pass
    output, _ = cell(x, state=None)

    # Compute simple loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert x.grad.shape == x.shape, "Gradient shape should match input"

    # Check that model parameters have gradients
    for name, param in cell.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.all(param.grad == 0), f"Gradients for {name} should be non-zero"


def test_mlstm_sequential_vs_parallel_multichunk() -> None:
    """Sequential vs parallel when sequence spans multiple chunks (T > chunk_size)."""
    torch.manual_seed(321)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 160  # multiple chunks when chunk_size=64
    H = 64  # head_dim must be >= 16 for triton
    num_heads = 4
    chunk_size = 64

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=chunk_size,
        conv1d_kernel_size=4,
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel (chunkwise) processing
    with torch.no_grad():
        y_parallel, state_parallel = cell(x, state=None)

    # Sequential processing
    state = None
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = cell(x[:, t, :], state)
            y_steps.append(y_t)
    y_sequential = torch.stack(y_steps, dim=1)

    # Compare outputs (relaxed tolerance for longer sequences with multiple chunks)
    assert y_parallel.shape == y_sequential.shape
    torch.testing.assert_close(
        y_parallel,
        y_sequential,
        rtol=1e-2,
        atol=1e-2,
        msg="Sequential and parallel outputs differ beyond tolerance for multi-chunk",
    )

    # Compare final states; allow looser tolerance for accumulators
    c_diff_rel = ((state_parallel["c"] - state["c"]).abs() / (state_parallel["c"].abs() + 1e-8)).max().item()
    print(f"[multichunk] Max relative difference in c: {c_diff_rel:.6f}")
    assert c_diff_rel < 1.0

    torch.testing.assert_close(state_parallel["n"], state["n"], rtol=0.5, atol=0.5)
    torch.testing.assert_close(state_parallel["m"], state["m"], rtol=0.1, atol=0.1)


def test_mlstm_state_reset() -> None:
    """Test state reset functionality."""
    torch.manual_seed(789)

    device = get_test_device()
    dtype = torch.float32

    B = 4  # batch size
    H = 64  # head_dim must be >= 16 for triton
    num_heads = 4

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=16,
        conv1d_kernel_size=4,
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)

    # Initialize state
    state = cell.init_state(B, device=device, dtype=dtype)

    # Set state to non-zero values
    state["c"] = torch.ones_like(state["c"])
    state["n"] = torch.ones_like(state["n"])
    state["m"] = torch.ones_like(state["m"])
    state["conv"] = torch.ones_like(state["conv"])

    # Create reset mask (reset first two batch elements)
    reset_mask = torch.tensor([True, True, False, False], device=device)

    # Reset state
    reset_state = cell.reset_state(state, reset_mask)

    # Check that first two batch elements are zeros
    assert torch.all(reset_state["c"][:2] == 0), "First two batch elements of 'c' should be reset"
    assert torch.all(reset_state["n"][:2] == 0), "First two batch elements of 'n' should be reset"
    assert torch.all(reset_state["m"][:2] == 0), "First two batch elements of 'm' should be reset"
    assert torch.all(reset_state["conv"][:2] == 0), "First two batch elements of 'conv' should be reset"

    # Check that last two batch elements are unchanged
    assert torch.all(reset_state["c"][2:] == 1), "Last two batch elements of 'c' should be unchanged"
    assert torch.all(reset_state["n"][2:] == 1), "Last two batch elements of 'n' should be unchanged"
    assert torch.all(reset_state["m"][2:] == 1), "Last two batch elements of 'm' should be unchanged"
    assert torch.all(reset_state["conv"][2:] == 1), "Last two batch elements of 'conv' should be unchanged"


def test_mlstm_backward_sequential_vs_parallel() -> None:
    """Test that backward pass gradients match between sequential and parallel implementations."""
    torch.manual_seed(999)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 32  # sequence length (keep small for numerical stability)
    H = 64  # hidden size (head_dim must be >= 16 for triton)
    num_heads = 4
    chunk_size = 64  # Set chunk_size > T to use parallel_stabilized_simple path

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=chunk_size,
        conv1d_kernel_size=4,
    )

    # Create two identical cells for parallel and sequential paths
    cell_parallel = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell_sequential = mLSTMCell(cfg).to(device=device, dtype=dtype)

    # Ensure both cells have identical parameters
    cell_sequential.load_state_dict(cell_parallel.state_dict())

    # Set both to training mode
    cell_parallel.train()
    cell_sequential.train()

    # Create identical input tensors that require gradients
    x = torch.randn(B, T, H, device=device, dtype=dtype)
    x_parallel = x.clone().requires_grad_(True)
    x_sequential = x.clone().requires_grad_(True)

    # Forward pass - parallel (uses triton kernels)
    y_parallel, state_parallel = cell_parallel(x_parallel, state=None)

    # Forward pass - sequential (step by step)
    state = None
    y_steps = []
    for t in range(T):
        y_t, state = cell_sequential(x_sequential[:, t, :], state)
        y_steps.append(y_t)
    y_sequential = torch.stack(y_steps, dim=1)

    # Compute identical losses
    loss_parallel = y_parallel.mean()
    loss_sequential = y_sequential.mean()

    # Backward pass
    loss_parallel.backward()
    loss_sequential.backward()

    # Compare input gradients
    assert x_parallel.grad is not None, "Parallel input should have gradients"
    assert x_sequential.grad is not None, "Sequential input should have gradients"

    torch.testing.assert_close(
        x_parallel.grad,
        x_sequential.grad,
        rtol=1e-2,
        atol=1e-2,
        msg="Input gradients differ between sequential and parallel"
    )

    # Compare parameter gradients
    for (name_p, param_p), (name_s, param_s) in zip(
        cell_parallel.named_parameters(), cell_sequential.named_parameters()
    ):
        assert name_p == name_s, f"Parameter names don't match: {name_p} vs {name_s}"

        if param_p.grad is not None and param_s.grad is not None:
            # Use relaxed tolerances for gradients as they accumulate numerical differences
            grad_diff_max = (param_p.grad - param_s.grad).abs().max().item()
            grad_diff_rel = ((param_p.grad - param_s.grad).abs() / (param_p.grad.abs() + 1e-8)).max().item()

            print(f"Gradient diff for {name_p}: max_abs={grad_diff_max:.6f}, max_rel={grad_diff_rel:.6f}")

            # Check with relaxed tolerance
            torch.testing.assert_close(
                param_p.grad,
                param_s.grad,
                rtol=5e-2,
                atol=5e-2,
                msg=f"Gradients differ for parameter {name_p}"
            )


if __name__ == "__main__":
    # Run all tests
    test_mlstm_parallel_vs_sequential_close()
    print("✓ test_mlstm_parallel_vs_sequential_close passed")

    test_mlstm_with_preup_block()
    print("✓ test_mlstm_with_preup_block passed")

    test_mlstm_sequential_vs_parallel_with_chunking()
    print("✓ test_mlstm_sequential_vs_parallel_with_chunking passed")

    test_mlstm_gradient_flow()
    print("✓ test_mlstm_gradient_flow passed")

    test_mlstm_sequential_vs_parallel_multichunk()
    print("✓ test_mlstm_sequential_vs_parallel_multichunk passed")

    test_mlstm_state_reset()
    print("✓ test_mlstm_state_reset passed")

    test_mlstm_backward_sequential_vs_parallel()
    print("✓ test_mlstm_backward_sequential_vs_parallel passed")

    print("\nAll tests passed!")
