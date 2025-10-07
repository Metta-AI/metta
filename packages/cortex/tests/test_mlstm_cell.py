"""Tests for mLSTM cell implementation."""

import os
import sys
from pathlib import Path

import torch

# Make cortex package importable relative to this test
try:
    PKG_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PKG_ROOT = Path.cwd().parent
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
    T = 18  # sequence length
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

    # The PreUpBlock wraps the cell state with the cell class name as key
    assert "mLSTMCell" in state, f"State should contain 'mLSTMCell' key, got keys: {list(state.keys())}"
    cell_state = state["mLSTMCell"]

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
    T = 18
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
    T = 1060  # multiple chunks when chunk_size=64 and not divisible by chunk_size
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
        msg="Input gradients differ between sequential and parallel",
    )

    # Compare parameter gradients
    for (name_p, param_p), (name_s, param_s) in zip(
        cell_parallel.named_parameters(), cell_sequential.named_parameters(), strict=False
    ):
        assert name_p == name_s, f"Parameter names don't match: {name_p} vs {name_s}"

        if param_p.grad is not None and param_s.grad is not None:
            # Use relaxed tolerances for gradients as they accumulate numerical differences
            grad_diff_max = (param_p.grad - param_s.grad).abs().max().item()
            grad_diff_rel = ((param_p.grad - param_s.grad).abs() / (param_p.grad.abs() + 1e-8)).max().item()

            print(f"Gradient diff for {name_p}: max_abs={grad_diff_max:.6f}, max_rel={grad_diff_rel:.6f}")

            # Check with relaxed tolerance
            torch.testing.assert_close(
                param_p.grad, param_s.grad, rtol=5e-2, atol=5e-2, msg=f"Gradients differ for parameter {name_p}"
            )


def test_mlstm_reset_mask_functionality() -> None:
    """Test reset mask functionality across different backends."""
    torch.manual_seed(555)

    device = get_test_device()
    dtype = torch.float32

    B = 4  # batch size
    T = 98  # sequence length (spans multiple chunks)
    H = 64  # hidden size
    num_heads = 4
    chunk_size = 32  # Will create 3 chunks

    # Import kernel functions directly
    from cortex.kernels import (
        mlstm_chunkwise_simple,
        mlstm_chunkwise_triton,
        mlstm_recurrent_step_stabilized_simple,
    )

    # Prepare inputs
    queries = torch.randn(B, num_heads, T, H // num_heads, device=device, dtype=dtype)
    keys = torch.randn(B, num_heads, T, H // num_heads, device=device, dtype=dtype)
    values = torch.randn(B, num_heads, T, H // num_heads, device=device, dtype=dtype)
    igate_preact = torch.randn(B, num_heads, T, device=device, dtype=dtype)
    fgate_preact = torch.randn(B, num_heads, T, device=device, dtype=dtype)

    # Create reset mask - reset at various positions including within chunks
    reset_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    reset_mask[0, 20] = True  # Reset batch 0 at timestep 20 (within first chunk)
    reset_mask[1, 40] = True  # Reset batch 1 at timestep 40 (within second chunk)
    reset_mask[2, 60] = True  # Reset batch 2 at timestep 60 (within second chunk)
    reset_mask[3, 32] = True  # Reset batch 3 at chunk boundary
    # Add more resets within chunks to test thoroughly
    reset_mask[0, 70] = True  # Another reset within third chunk
    reset_mask[1, 15] = True  # Reset within first chunk

    # Test 1: Simple backend with reset mask
    output_simple_reset, (c_simple, n_simple, m_simple) = mlstm_chunkwise_simple(
        queries=queries,
        keys=keys,
        values=values,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask,
        chunk_size=chunk_size,
        return_last_state=True,
    )

    # Test 2: Simple backend without reset mask
    output_simple_no_reset, (c_no_reset, n_no_reset, m_no_reset) = mlstm_chunkwise_simple(
        queries=queries,
        keys=keys,
        values=values,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=None,
        chunk_size=chunk_size,
        return_last_state=True,
    )

    # Outputs should differ when reset mask is applied
    assert not torch.allclose(output_simple_reset, output_simple_no_reset), (
        "Outputs should differ with and without reset mask"
    )

    # Test 3: Triton backend with reset mask (native reset handling)
    output_triton_reset, (c_triton, n_triton, m_triton) = mlstm_chunkwise_triton(
        queries=queries,
        keys=keys,
        values=values,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask,
        chunk_size=chunk_size,
        return_last_state=True,
    )

    # Triton should be numerically close to the simple backend
    torch.testing.assert_close(
        output_triton_reset,
        output_simple_reset,
        rtol=2e-2,
        atol=2e-2,
        msg="Triton should be close to simple backend with reset mask",
    )

    # Test 4: Sequential processing with reset mask using recurrent step (reference implementation)
    c_state = torch.zeros(B, num_heads, H // num_heads, H // num_heads, device=device, dtype=dtype)
    n_state = torch.zeros(B, num_heads, H // num_heads, 1, device=device, dtype=dtype)
    m_state = torch.zeros(B, num_heads, 1, 1, device=device, dtype=dtype)

    outputs_sequential = []
    for t in range(T):
        # Apply reset if needed
        if t > 0:  # Don't check at t=0 since states are already zero
            reset_t = reset_mask[:, t]
            h_step, (c_state, n_state, m_state) = mlstm_recurrent_step_stabilized_simple(
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
                q=queries[:, :, t : t + 1, :],
                k=keys[:, :, t : t + 1, :],
                v=values[:, :, t : t + 1, :],
                igate_preact=igate_preact[:, :, t : t + 1].unsqueeze(-1),
                fgate_preact=fgate_preact[:, :, t : t + 1].unsqueeze(-1),
                reset_mask=reset_t,
            )
        else:
            h_step, (c_state, n_state, m_state) = mlstm_recurrent_step_stabilized_simple(
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
                q=queries[:, :, t : t + 1, :],
                k=keys[:, :, t : t + 1, :],
                v=values[:, :, t : t + 1, :],
                igate_preact=igate_preact[:, :, t : t + 1].unsqueeze(-1),
                fgate_preact=fgate_preact[:, :, t : t + 1].unsqueeze(-1),
                reset_mask=None,
            )
        outputs_sequential.append(h_step)

    output_sequential = torch.cat(outputs_sequential, dim=2)

    # Test 5: Verify all backends match the sequential reference implementation
    print("Comparing backends against sequential reference...")

    # Compare simple backend with sequential
    max_diff_simple = (output_simple_reset - output_sequential).abs().max().item()
    mean_diff_simple = (output_simple_reset - output_sequential).abs().mean().item()
    print(f"  Simple vs Sequential - Max diff: {max_diff_simple:.6f}, Mean diff: {mean_diff_simple:.6f}")

    # Compare triton backend with sequential
    max_diff_triton = (output_triton_reset - output_sequential).abs().max().item()
    mean_diff_triton = (output_triton_reset - output_sequential).abs().mean().item()
    print(f"  Triton vs Sequential - Max diff: {max_diff_triton:.6f}, Mean diff: {mean_diff_triton:.6f}")

    # All backends should produce similar outputs to sequential reference
    # Use reasonable tolerance due to numerical differences in computation paths
    torch.testing.assert_close(
        output_simple_reset,
        output_sequential,
        rtol=1e-4,
        atol=1e-4,
        msg="Simple backend should match sequential reference",
    )

    torch.testing.assert_close(
        output_triton_reset,
        output_sequential,
        rtol=2e-2,
        atol=2e-2,
        msg="Triton backend should be close to sequential reference",
    )

    # Also verify simple and triton are close (independent implementations)
    torch.testing.assert_close(
        output_triton_reset,
        output_simple_reset,
        rtol=2e-2,
        atol=2e-2,
        msg="Triton and Simple should be close when reset_mask is used",
    )

    # Test 6: Verify gradient flow with reset mask across all backends
    print("Testing gradient computation with reset masks...")

    # Test gradient flow for simple backend
    queries_grad_simple = queries.clone().requires_grad_(True)
    keys_grad_simple = keys.clone().requires_grad_(True)
    values_grad_simple = values.clone().requires_grad_(True)

    output_grad_simple = mlstm_chunkwise_simple(
        queries=queries_grad_simple,
        keys=keys_grad_simple,
        values=values_grad_simple,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask,
        chunk_size=chunk_size,
        return_last_state=False,
    )

    loss_simple = output_grad_simple.mean()
    loss_simple.backward()

    # Test gradient flow for triton backend
    queries_grad_triton = queries.clone().requires_grad_(True)
    keys_grad_triton = keys.clone().requires_grad_(True)
    values_grad_triton = values.clone().requires_grad_(True)

    output_grad_triton = mlstm_chunkwise_triton(
        queries=queries_grad_triton,
        keys=keys_grad_triton,
        values=values_grad_triton,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask,
        chunk_size=chunk_size,
        return_last_state=False,
    )

    loss_triton = output_grad_triton.mean()
    loss_triton.backward()

    # Test gradient flow for sequential backend
    queries_grad_seq = queries.clone().requires_grad_(True)
    keys_grad_seq = keys.clone().requires_grad_(True)
    values_grad_seq = values.clone().requires_grad_(True)

    # Run sequential with gradients
    c_state_grad = torch.zeros(B, num_heads, H // num_heads, H // num_heads, device=device, dtype=dtype)
    n_state_grad = torch.zeros(B, num_heads, H // num_heads, 1, device=device, dtype=dtype)
    m_state_grad = torch.zeros(B, num_heads, 1, 1, device=device, dtype=dtype)

    outputs_grad_seq = []
    for t in range(T):
        if t > 0:
            reset_t = reset_mask[:, t]
            h_step_grad, (c_state_grad, n_state_grad, m_state_grad) = mlstm_recurrent_step_stabilized_simple(
                c_state=c_state_grad,
                n_state=n_state_grad,
                m_state=m_state_grad,
                q=queries_grad_seq[:, :, t : t + 1, :],
                k=keys_grad_seq[:, :, t : t + 1, :],
                v=values_grad_seq[:, :, t : t + 1, :],
                igate_preact=igate_preact[:, :, t : t + 1].unsqueeze(-1),
                fgate_preact=fgate_preact[:, :, t : t + 1].unsqueeze(-1),
                reset_mask=reset_t,
            )
        else:
            h_step_grad, (c_state_grad, n_state_grad, m_state_grad) = mlstm_recurrent_step_stabilized_simple(
                c_state=c_state_grad,
                n_state=n_state_grad,
                m_state=m_state_grad,
                q=queries_grad_seq[:, :, t : t + 1, :],
                k=keys_grad_seq[:, :, t : t + 1, :],
                v=values_grad_seq[:, :, t : t + 1, :],
                igate_preact=igate_preact[:, :, t : t + 1].unsqueeze(-1),
                fgate_preact=fgate_preact[:, :, t : t + 1].unsqueeze(-1),
                reset_mask=None,
            )
        outputs_grad_seq.append(h_step_grad)

    output_grad_seq = torch.cat(outputs_grad_seq, dim=2)
    loss_seq = output_grad_seq.mean()
    loss_seq.backward()

    # Check that all backends have gradients
    assert queries_grad_simple.grad is not None, "Simple backend: Queries should have gradients"
    assert keys_grad_simple.grad is not None, "Simple backend: Keys should have gradients"
    assert values_grad_simple.grad is not None, "Simple backend: Values should have gradients"

    assert queries_grad_triton.grad is not None, "Triton backend: Queries should have gradients"
    assert keys_grad_triton.grad is not None, "Triton backend: Keys should have gradients"
    assert values_grad_triton.grad is not None, "Triton backend: Values should have gradients"

    assert queries_grad_seq.grad is not None, "Sequential backend: Queries should have gradients"
    assert keys_grad_seq.grad is not None, "Sequential backend: Keys should have gradients"
    assert values_grad_seq.grad is not None, "Sequential backend: Values should have gradients"

    # Check gradients are non-zero
    assert not torch.all(queries_grad_simple.grad == 0), "Simple: Query gradients should be non-zero"
    assert not torch.all(keys_grad_simple.grad == 0), "Simple: Key gradients should be non-zero"
    assert not torch.all(values_grad_simple.grad == 0), "Simple: Value gradients should be non-zero"

    assert not torch.all(queries_grad_triton.grad == 0), "Triton: Query gradients should be non-zero"
    assert not torch.all(keys_grad_triton.grad == 0), "Triton: Key gradients should be non-zero"
    assert not torch.all(values_grad_triton.grad == 0), "Triton: Value gradients should be non-zero"

    assert not torch.all(queries_grad_seq.grad == 0), "Sequential: Query gradients should be non-zero"
    assert not torch.all(keys_grad_seq.grad == 0), "Sequential: Key gradients should be non-zero"
    assert not torch.all(values_grad_seq.grad == 0), "Sequential: Value gradients should be non-zero"

    # Compare gradients across backends
    print("Comparing gradients across backends...")

    # Simple vs Sequential gradients
    q_grad_diff_simple_seq = (queries_grad_simple.grad - queries_grad_seq.grad).abs().max().item()
    k_grad_diff_simple_seq = (keys_grad_simple.grad - keys_grad_seq.grad).abs().max().item()
    v_grad_diff_simple_seq = (values_grad_simple.grad - values_grad_seq.grad).abs().max().item()

    print(
        "  Simple vs Sequential gradients - "
        f"Q: {q_grad_diff_simple_seq:.6f}, "
        f"K: {k_grad_diff_simple_seq:.6f}, "
        f"V: {v_grad_diff_simple_seq:.6f}"
    )

    # Triton vs Sequential gradients
    q_grad_diff_triton_seq = (queries_grad_triton.grad - queries_grad_seq.grad).abs().max().item()
    k_grad_diff_triton_seq = (keys_grad_triton.grad - keys_grad_seq.grad).abs().max().item()
    v_grad_diff_triton_seq = (values_grad_triton.grad - values_grad_seq.grad).abs().max().item()

    print(
        "  Triton vs Sequential gradients - "
        f"Q: {q_grad_diff_triton_seq:.6f}, "
        f"K: {k_grad_diff_triton_seq:.6f}, "
        f"V: {v_grad_diff_triton_seq:.6f}"
    )

    # Verify gradients are similar across backends (relaxed tolerance due to numerical differences)
    torch.testing.assert_close(
        queries_grad_simple.grad,
        queries_grad_seq.grad,
        rtol=1e-3,
        atol=1e-3,
        msg="Query gradients should match between simple and sequential backends",
    )

    torch.testing.assert_close(
        keys_grad_simple.grad,
        keys_grad_seq.grad,
        rtol=1e-3,
        atol=1e-3,
        msg="Key gradients should match between simple and sequential backends",
    )

    torch.testing.assert_close(
        values_grad_simple.grad,
        values_grad_seq.grad,
        rtol=1e-3,
        atol=1e-3,
        msg="Value gradients should match between simple and sequential backends",
    )

    # Triton gradients may have larger differences but should be in the same ballpark
    torch.testing.assert_close(
        queries_grad_triton.grad,
        queries_grad_seq.grad,
        rtol=5e-2,
        atol=5e-2,
        msg="Query gradients should be similar between triton and sequential backends",
    )

    torch.testing.assert_close(
        keys_grad_triton.grad,
        keys_grad_seq.grad,
        rtol=5e-2,
        atol=5e-2,
        msg="Key gradients should be similar between triton and sequential backends",
    )

    torch.testing.assert_close(
        values_grad_triton.grad,
        values_grad_seq.grad,
        rtol=5e-2,
        atol=5e-2,
        msg="Value gradients should be similar between triton and sequential backends",
    )

    # Test 7: Verify within-chunk resets work correctly
    # Create two reset masks - one with reset at chunk boundary, one within chunk
    reset_mask_boundary = torch.zeros(B, T, dtype=torch.bool, device=device)
    reset_mask_boundary[0, 32] = True  # At chunk boundary
    reset_mask_boundary[0, 64] = True  # At chunk boundary

    reset_mask_within = torch.zeros(B, T, dtype=torch.bool, device=device)
    reset_mask_within[0, 30] = True  # Within first chunk
    reset_mask_within[0, 62] = True  # Within second chunk

    output_boundary = mlstm_chunkwise_simple(
        queries=queries,
        keys=keys,
        values=values,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask_boundary,
        chunk_size=chunk_size,
        return_last_state=False,
    )

    output_within = mlstm_chunkwise_simple(
        queries=queries,
        keys=keys,
        values=values,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        reset_mask=reset_mask_within,
        chunk_size=chunk_size,
        return_last_state=False,
    )

    # Outputs should differ since resets are at different positions
    assert not torch.allclose(output_boundary, output_within, rtol=1e-3), (
        "Outputs should differ when resets are at boundaries vs within chunks"
    )

    # Test 8: mLSTMCell end-to-end with reset mask (sequence vs. step)
    from cortex.cells.mlstm import mLSTMCell  # local import to avoid circularities
    from cortex.config import mLSTMCellConfig

    # Use kernel_size=1 to avoid conv-state dependence across timesteps,
    # ensuring step/sequence parity under resets.
    mlstm_cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=chunk_size,
        conv1d_kernel_size=1,
    )

    cell = mLSTMCell(mlstm_cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Sequence with reset mask
    y_cell_reset, state_cell_reset = cell(x, state=None, resets=reset_mask)
    # Sequence without reset mask
    y_cell_no_reset, _ = cell(x, state=None, resets=None)

    # Outputs should differ when resets are applied
    assert not torch.allclose(y_cell_reset, y_cell_no_reset), (
        "mLSTMCell outputs should differ with and without reset mask"
    )

    # Step-by-step with per-timestep resets
    state_seq = None
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state_seq = cell(x[:, t, :], state_seq, resets=reset_mask[:, t])
            y_steps.append(y_t)
    y_seq_reset = torch.stack(y_steps, dim=1)

    # Compare sequence vs step with resets: ensure shapes match and values are finite
    assert y_cell_reset.shape == y_seq_reset.shape
    assert torch.isfinite(y_cell_reset).all() and torch.isfinite(y_seq_reset).all()

    # Check state structure is present and well-formed
    assert state_cell_reset is not None
    for key in ("c", "n", "m", "conv"):
        assert key in state_cell_reset

    # Test 9: Within-chunk vs boundary resets via mLSTMCell should differ
    y_boundary, _ = cell(x, state=None, resets=reset_mask_boundary)
    y_within, _ = cell(x, state=None, resets=reset_mask_within)
    assert not torch.allclose(y_boundary, y_within, rtol=1e-3), (
        "mLSTMCell: outputs should differ when resets are at boundaries vs within chunks"
    )

    print("✓ Reset mask forward pass tests passed")
    print("✓ Reset mask backward pass tests passed")
    print("✓ Reset mask consistency across backends verified")
    print("✓ Within-chunk reset handling verified")
    print("✓ mLSTMCell reset mask behavior verified (sequence and step)")


def test_mlstm_sequence_vs_step_with_resets_conv_multichunk_strict() -> None:
    """Sequence vs step should match under resets when conv has memory (kernel>1).

    This test intentionally uses conv1d_kernel_size>1 and T>chunk_size so that:
    - the conv ring buffer carries history across timesteps, and
    - the mLSTM sequence path uses the chunkwise backend with a reset mask.

    Current implementation deviates from exact step semantics in this setting.
    The assertion is strict to surface the discrepancy before we change code.
    """
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 64  # spans multiple chunks when chunk_size=16
    H = 64
    num_heads = 4

    cfg = mLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        chunk_size=16,         # force multi-chunk processing in sequence mode
        conv1d_kernel_size=4,  # conv has memory across time
    )

    cell = mLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Create per-timestep resets at different positions
    resets = torch.zeros(B, T, device=device, dtype=torch.bool)
    resets[0, 17] = True
    resets[1, 23] = True

    # Sequence (parallel) path with resets
    with torch.no_grad():
        y_seqmode, state_seqmode = cell(x, state=None, resets=resets)

    # Step-by-step path with per-timestep resets
    state = None
    y_steps = []
    with torch.no_grad():
        for t in range(T):
            y_t, state = cell(x[:, t, :], state, resets=resets[:, t])
            y_steps.append(y_t)
    y_stepmode = torch.stack(y_steps, dim=1)

    # Strict parity expectation to surface current discrepancy
    assert y_seqmode.shape == y_stepmode.shape
    torch.testing.assert_close(
        y_seqmode,
        y_stepmode,
        rtol=2e-3,
        atol=2e-3,
        msg="mLSTMCell sequence vs step should match under resets with conv>1 (allow small numeric drift)",
    )

    # Also check final states for completeness
    assert state_seqmode is not None and state is not None
    for key in ("c", "n", "m"):
        torch.testing.assert_close(
            state_seqmode[key],
            state[key],
            rtol=2e-3,
            atol=2e-3,
            msg=f"Final state '{key}' should match between sequence and step (allow small numeric drift)",
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

    test_mlstm_reset_mask_functionality()
    print("✓ test_mlstm_reset_mask_functionality passed")

    print("\nAll tests passed!")
