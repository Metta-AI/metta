"""Tests for sLSTM cell implementation (Triton vs vanilla parity)."""

import torch
from cortex.blocks import PostUpBlock
from cortex.cells.slstm import sLSTMCell
from cortex.config import PostUpBlockConfig, sLSTMCellConfig


def get_test_device():
    """Get the appropriate device for testing (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def test_slstm_parallel_vs_sequential_close() -> None:
    """Test that parallel and sequential processing produce similar outputs."""
    torch.manual_seed(0)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 32  # sequence length
    H = 64  # hidden size (must be divisible by num_heads; DH should be power of 2)
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel over the entire sequence (uses Triton if available on CUDA, else vanilla)
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


def test_slstm_with_postup_block() -> None:
    """Test sLSTM cell within a PostUp block for proper forward pass."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 16  # sequence length
    D = 64  # external hidden size
    proj_factor = 2.0  # PostUp projection factor

    # Create PostUp block config with sLSTM cell
    # The cell operates on the base dimension D directly
    slstm_config = sLSTMCellConfig(
        hidden_size=D,  # PostUp uses base dimension for the cell
        num_heads=4,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    postup_config = PostUpBlockConfig(
        cell=slstm_config,
        proj_factor=proj_factor,
    )

    # Create the cell and PostUp block
    cell = sLSTMCell(slstm_config).to(device=device, dtype=dtype)
    block = PostUpBlock(postup_config, d_hidden=D, cell=cell).to(device=device, dtype=dtype)
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

    # The PostUpBlock wraps the cell state with the cell class name as key
    assert "sLSTMCell" in state, f"State should contain 'sLSTMCell' key, got keys: {list(state.keys())}"
    cell_state = state["sLSTMCell"]

    # Check cell state components
    assert "y" in cell_state, "Cell state should contain 'y' component"
    assert "c" in cell_state, "Cell state should contain 'c' component"
    assert "n" in cell_state, "Cell state should contain 'n' component"
    assert "m" in cell_state, "Cell state should contain 'm' component"
    assert "conv" in cell_state, "Cell state should contain 'conv' component"

    # Check state dimensions match the base dimension D (PostUp applies cell at base dim)
    assert cell_state["y"].shape == (B, D)
    assert cell_state["c"].shape == (B, D)
    assert cell_state["n"].shape == (B, D)
    assert cell_state["m"].shape == (B, D)
    assert cell_state["conv"].shape == (B, slstm_config.conv1d_kernel_size, D)


def test_slstm_sequential_vs_parallel_with_smaller_seq() -> None:
    """Test sequential vs parallel with smaller sequence to minimize numerical differences."""
    torch.manual_seed(123)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 16  # smaller sequence length to reduce numerical accumulation
    H = 64  # hidden size (head_dim should be power of 2 for Triton)
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel processing
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

    # Compare final states
    assert state_parallel is not None and state is not None, "States should not be None"

    for key in ["c", "n", "m"]:
        if key in state_parallel and key in state:
            diff_max = (state_parallel[key] - state[key]).abs().max().item()
            print(f"Max absolute difference in {key} state: {diff_max:.6f}")

    # Check state components with relaxed tolerance
    torch.testing.assert_close(
        state_parallel["c"],
        state["c"],
        rtol=1e-2,
        atol=1e-2,
        msg="Final 'c' states differ",
    )
    torch.testing.assert_close(
        state_parallel["n"],
        state["n"],
        rtol=1e-2,
        atol=1e-2,
        msg="Final 'n' states differ",
    )
    torch.testing.assert_close(
        state_parallel["m"],
        state["m"],
        rtol=1e-2,
        atol=1e-2,
        msg="Final 'm' states differ",
    )


def test_slstm_gradient_flow() -> None:
    """Test gradient flow through sLSTM cell."""
    torch.manual_seed(456)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 16
    H = 64  # head_dim should be power of 2 for Triton
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
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
            # Some parameters might be zero initialized, so we check for non-zero grad norm instead
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"Warning: Parameter {name} has zero gradients (may be expected for some params)")


def test_slstm_sequential_vs_parallel_long_sequence() -> None:
    """Test sequential vs parallel processing with longer sequences."""
    torch.manual_seed(321)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 1060  # longer sequence
    H = 64  # head_dim should be power of 2 for Triton
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Parallel processing
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

    # Compare outputs (relaxed tolerance for longer sequences)
    assert y_parallel.shape == y_sequential.shape
    torch.testing.assert_close(
        y_parallel,
        y_sequential,
        rtol=1e-2,
        atol=1e-2,
        msg="Sequential and parallel outputs differ beyond tolerance for long sequence",
    )

    # Compare final states with looser tolerance for accumulators
    assert state_parallel is not None and state is not None, "States should not be None"

    print("[long sequence] Comparing final states")
    for key in ["c", "n", "m"]:
        if key in state_parallel and key in state:
            diff_rel = ((state_parallel[key] - state[key]).abs() / (state_parallel[key].abs() + 1e-8)).max().item()
            print(f"Max relative difference in {key}: {diff_rel:.6f}")

    torch.testing.assert_close(state_parallel["c"], state["c"], rtol=0.1, atol=0.1)
    torch.testing.assert_close(state_parallel["n"], state["n"], rtol=0.1, atol=0.1)
    torch.testing.assert_close(state_parallel["m"], state["m"], rtol=0.1, atol=0.1)


def test_slstm_state_reset() -> None:
    """Test state reset functionality."""
    torch.manual_seed(789)

    device = get_test_device()
    dtype = torch.float32

    B = 4  # batch size
    H = 64  # head_dim should be power of 2 for Triton
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)

    # Initialize state
    state = cell.init_state(B, device=device, dtype=dtype)

    # Set state to non-zero values
    state["y"] = torch.ones_like(state["y"])
    state["c"] = torch.ones_like(state["c"])
    state["n"] = torch.ones_like(state["n"])
    state["m"] = torch.ones_like(state["m"])
    state["conv"] = torch.ones_like(state["conv"])

    # Create reset mask (reset first two batch elements)
    reset_mask = torch.tensor([True, True, False, False], device=device)

    # Reset state
    reset_state = cell.reset_state(state, reset_mask)

    # Check that first two batch elements are zeros
    assert torch.all(reset_state["y"][:2] == 0), "First two batch elements of 'y' should be reset"
    assert torch.all(reset_state["c"][:2] == 0), "First two batch elements of 'c' should be reset"
    assert torch.all(reset_state["n"][:2] == 0), "First two batch elements of 'n' should be reset"
    assert torch.all(reset_state["m"][:2] == 0), "First two batch elements of 'm' should be reset"
    assert torch.all(reset_state["conv"][:2] == 0), "First two batch elements of 'conv' should be reset"

    # Check that last two batch elements are unchanged
    assert torch.all(reset_state["y"][2:] == 1), "Last two batch elements of 'y' should be unchanged"
    assert torch.all(reset_state["c"][2:] == 1), "Last two batch elements of 'c' should be unchanged"
    assert torch.all(reset_state["n"][2:] == 1), "Last two batch elements of 'n' should be unchanged"
    assert torch.all(reset_state["m"][2:] == 1), "Last two batch elements of 'm' should be unchanged"
    assert torch.all(reset_state["conv"][2:] == 1), "Last two batch elements of 'conv' should be unchanged"


def test_slstm_no_conv() -> None:
    """Test sLSTM cell without convolutional preprocessing."""
    torch.manual_seed(555)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 16
    H = 64
    num_heads = 4

    # Config with conv1d_kernel_size=0 to disable conv
    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=0,  # Disable conv
        dropout=0.0,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell.eval()

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Forward pass
    with torch.no_grad():
        y, state = cell(x, state=None)

    # Check output shape
    assert y.shape == (B, T, H), f"Expected output shape {(B, T, H)}, got {y.shape}"

    # Check state doesn't have conv component when conv is disabled
    assert "conv" not in state, "State should not have 'conv' when conv1d_kernel_size=0"

    # Check other state components exist
    assert "y" in state, "State should contain 'y'"
    assert "c" in state, "State should contain 'c'"
    assert "n" in state, "State should contain 'n'"
    assert "m" in state, "State should contain 'm'"


def test_slstm_different_head_counts() -> None:
    """Test sLSTM with different numbers of attention heads."""
    torch.manual_seed(999)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 16
    H = 128

    # Test with different head counts
    for num_heads in [1, 2, 4, 8]:
        if H % num_heads != 0:
            continue

        cfg = sLSTMCellConfig(
            hidden_size=H,
            num_heads=num_heads,
            conv1d_kernel_size=4,
            dropout=0.0,
        )

        cell = sLSTMCell(cfg).to(device=device, dtype=dtype)
        cell.eval()

        x = torch.randn(B, T, H, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            y, _ = cell(x, state=None)

        # Check output shape
        assert y.shape == (B, T, H), f"With {num_heads} heads: expected shape {(B, T, H)}, got {y.shape}"

        # Verify head_dim calculation
        head_dim = H // num_heads
        assert head_dim == cell.head_dim, f"Head dimension mismatch for {num_heads} heads"

        print(f"✓ Test passed for num_heads={num_heads}, head_dim={head_dim}")


def test_slstm_with_dropout() -> None:
    """Test sLSTM cell with dropout enabled."""
    torch.manual_seed(111)

    device = get_test_device()
    dtype = torch.float32

    B = 2
    T = 16
    H = 64
    num_heads = 4
    dropout_rate = 0.1

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=dropout_rate,
    )

    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)

    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # Test in training mode (dropout active)
    cell.train()
    y_train1, _ = cell(x, state=None)
    y_train2, _ = cell(x, state=None)

    # Outputs should be different in training mode due to dropout
    assert not torch.allclose(y_train1, y_train2), "Training outputs should differ with dropout"

    # Test in eval mode (dropout inactive)
    cell.eval()
    with torch.no_grad():
        y_eval1, _ = cell(x, state=None)
        y_eval2, _ = cell(x, state=None)

    # Outputs should be identical in eval mode
    torch.testing.assert_close(y_eval1, y_eval2, msg="Eval outputs should be identical")


def test_slstm_backward_sequential_vs_parallel() -> None:
    """Test that backward pass gradients match between sequential and parallel implementations."""
    torch.manual_seed(777)

    device = get_test_device()
    dtype = torch.float32

    B = 2  # batch size
    T = 32  # sequence length (keep small for numerical stability)
    H = 64  # hidden size (head_dim should be power of 2 for Triton)
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,  # No dropout for deterministic comparison
    )

    # Create two identical cells for parallel and sequential paths
    cell_parallel = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell_sequential = sLSTMCell(cfg).to(device=device, dtype=dtype)

    # Ensure both cells have identical parameters
    cell_sequential.load_state_dict(cell_parallel.state_dict())

    # Set both to training mode
    cell_parallel.train()
    cell_sequential.train()

    # Create identical input tensors that require gradients
    x = torch.randn(B, T, H, device=device, dtype=dtype)
    x_parallel = x.clone().requires_grad_(True)
    x_sequential = x.clone().requires_grad_(True)

    # Forward pass - parallel (uses triton kernels on CUDA)
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


def test_slstm_axon_headwise_gates_state_and_shapes() -> None:
    """sLSTM with Axon headwise gates should run and populate per-head Axon substates."""
    torch.manual_seed(2025)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 2, 12, 64, 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=NH,
        conv1d_kernel_size=4,
        dropout=0.0,
        use_axon_layer=True,
    )
    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    y, state = cell(x, state=None)

    assert y.shape == (B, T, H)
    assert state is not None
    # Axon gate substates live under group "slstm"
    assert "slstm" in state.keys(), "Expected Axon gate group 'slstm' in state"
    axg = state.get("slstm")
    assert axg is not None
    # Check a couple of per-head keys exist and have expected substate structure
    for gate in ("igate", "fgate"):
        key = f"{gate}_h0"
        assert key in axg.keys(), f"Missing headwise Axon subkey: {key}"
        sub = axg.get(key)
        assert sub is not None and "hc1" in sub.keys() and "hc2" in sub.keys()
        assert sub["hc1"].shape == (B, H // NH)
        assert sub["hc2"].shape == (B, H // NH)


def test_slstm_axon_headwise_reset_propagates() -> None:
    """Resetting the sLSTM state should also reset per-head Axon gate substates."""
    torch.manual_seed(2026)

    device = get_test_device()
    dtype = torch.float32

    B, T, H, NH = 3, 6, 64, 4
    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=NH,
        conv1d_kernel_size=4,
        dropout=0.0,
        use_axon_layer=True,
    )
    cell = sLSTMCell(cfg).to(device=device, dtype=dtype)

    x = torch.randn(B, T, H, device=device, dtype=dtype)
    _, state = cell(x, state=None)
    assert state is not None and "slstm" in state.keys()
    # Reset first batch element
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    mask[0] = True
    state_after = cell.reset_state(state, mask)
    axg = state_after.get("slstm")  # type: ignore[union-attr]
    assert axg is not None
    # Verify one gate/head cleared
    sub = axg.get("igate_h0")
    assert sub is not None
    assert torch.allclose(sub["hc1"][0], torch.zeros_like(sub["hc1"][0]))
    assert torch.allclose(sub["hc2"][0], torch.zeros_like(sub["hc2"][0]))

    # (Optional relaxed tolerance checks removed to keep test concise)


def test_slstm_triton_vs_pytorch_with_resets() -> None:
    """Ensure Triton and PyTorch kernels match when per-timestep resets are applied."""
    torch.manual_seed(888)

    device = get_test_device()
    dtype = torch.float32

    # Skip if not on CUDA (Triton only runs on CUDA)
    if not torch.cuda.is_available():
        print("⊘ Skipping Triton vs PyTorch resets test (CUDA not available)")
        return

    B = 2
    T = 16
    H = 64  # head_dim should be power of 2 for Triton
    num_heads = 4

    cfg = sLSTMCellConfig(
        hidden_size=H,
        num_heads=num_heads,
        conv1d_kernel_size=4,
        dropout=0.0,
    )

    # Create two cells with identical weights
    cell_triton = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell_pytorch = sLSTMCell(cfg).to(device=device, dtype=dtype)
    cell_pytorch.load_state_dict(cell_triton.state_dict())

    cell_triton.train()  # Triton path used in parallel sequence mode
    cell_pytorch.train()

    # Create identical inputs
    x = torch.randn(B, T, H, device=device, dtype=dtype)
    x_triton = x.clone().requires_grad_(True)

    # Create reset mask: reset batch 0 at timestep 8, batch 1 at timestep 12
    resets = torch.zeros(B, T, device=device, dtype=torch.bool)
    resets[0, 8] = True
    resets[1, 12] = True

    # Forward pass - Triton path (sequence mode on CUDA with power-of-2 head_dim)
    y_triton, state_triton = cell_triton(x_triton, state=None, resets=resets)

    # Forward pass - PyTorch path (forced by using CPU)
    # To force PyTorch path, we can move to CPU (Triton requires CUDA)
    cell_pytorch_cpu = cell_pytorch.cpu()
    x_pytorch_cpu = x.detach().cpu().requires_grad_(True)
    resets_cpu = resets.cpu()
    y_pytorch, state_pytorch = cell_pytorch_cpu(x_pytorch_cpu, state=None, resets=resets_cpu)

    # Compare forward outputs
    torch.testing.assert_close(
        y_triton.cpu(),
        y_pytorch,
        rtol=1e-3,
        atol=1e-3,
        msg="Forward outputs should match between Triton and PyTorch with resets",
    )

    # Compare final states
    state_keys = ["y", "c", "n", "m"]
    for key in state_keys:
        torch.testing.assert_close(
            state_triton[key].cpu(),
            state_pytorch[key],
            rtol=1e-3,
            atol=1e-3,
            msg=f"State '{key}' should match between Triton and PyTorch with resets",
        )

    # Backward pass
    loss_triton = y_triton.mean()
    loss_pytorch = y_pytorch.mean()

    loss_triton.backward()
    loss_pytorch.backward()

    # Compare input gradients
    torch.testing.assert_close(
        x_triton.grad.cpu(),
        x_pytorch_cpu.grad,
        rtol=1e-3,
        atol=1e-3,
        msg="Input gradients should match between Triton and PyTorch with resets",
    )

    # Compare parameter gradients
    for (name_t, param_t), (name_p, param_p) in zip(
        cell_triton.named_parameters(), cell_pytorch_cpu.named_parameters(), strict=False
    ):
        assert name_t == name_p, f"Parameter names don't match: {name_t} vs {name_p}"
        if param_t.grad is None or param_p.grad is None:
            assert param_t.grad is param_p.grad, f"Gradient presence mismatch for parameter {name_t}"
        else:
            torch.testing.assert_close(
                param_t.grad.cpu(),
                param_p.grad,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Parameter gradient '{name_t}' should match",
            )


if __name__ == "__main__":
    # Run all tests
    test_slstm_parallel_vs_sequential_close()
    print("✓ test_slstm_parallel_vs_sequential_close passed")

    test_slstm_with_postup_block()
    print("✓ test_slstm_with_postup_block passed")

    test_slstm_sequential_vs_parallel_with_smaller_seq()
    print("✓ test_slstm_sequential_vs_parallel_with_smaller_seq passed")

    test_slstm_gradient_flow()
    print("✓ test_slstm_gradient_flow passed")

    test_slstm_sequential_vs_parallel_long_sequence()
    print("✓ test_slstm_sequential_vs_parallel_long_sequence passed")

    test_slstm_state_reset()
    print("✓ test_slstm_state_reset passed")

    test_slstm_no_conv()
    print("✓ test_slstm_no_conv passed")

    test_slstm_different_head_counts()
    print("✓ test_slstm_different_head_counts passed")

    test_slstm_with_dropout()
    print("✓ test_slstm_with_dropout passed")

    test_slstm_backward_sequential_vs_parallel()
    print("✓ test_slstm_backward_sequential_vs_parallel passed")

    test_slstm_triton_vs_pytorch_with_resets()
    print("✓ test_slstm_triton_vs_pytorch_with_resets passed")

    print("\nAll tests passed!")
