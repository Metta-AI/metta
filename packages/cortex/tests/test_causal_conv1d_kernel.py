"""Tests for causal conv1d kernel implementations.

This test compares the Triton kernel implementation against the pure PyTorch
reference implementation, verifying both forward and backward passes.
"""

import pytest
import torch
from cortex.kernels import causal_conv1d_pytorch, causal_conv1d_triton
from cortex.utils import TRITON_AVAILABLE


def get_test_device():
    """Get the appropriate device for testing (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
def test_causal_conv1d_triton_vs_pytorch_forward():
    """Test that Triton and PyTorch implementations produce similar forward outputs."""
    torch.manual_seed(42)

    device = get_test_device()
    if not device.type == "cuda":
        pytest.skip("Triton kernel requires CUDA")

    dtype = torch.float32
    B = 2  # batch size
    T = 64  # sequence length
    F = 32  # feature dimension
    KS = 7  # kernel size

    # Create inputs - channel-mixing mode (groups=1)
    x = torch.randn(B, T, F, device=device, dtype=dtype)
    weight = torch.randn(F, F, KS, device=device, dtype=dtype)
    bias = torch.randn(F, device=device, dtype=dtype)
    conv_state = torch.randn(B, KS, F, device=device, dtype=dtype)

    # Create reset mask with resets at various timesteps
    resets = torch.zeros(B, T, dtype=torch.bool, device=device)
    resets[0, 15] = True  # Reset batch 0 at timestep 15
    resets[1, 30] = True  # Reset batch 1 at timestep 30
    resets[0, 45] = True  # Another reset for batch 0

    # Test PyTorch implementation (reference)
    y_pytorch, state_pytorch = causal_conv1d_pytorch(
        conv_state=conv_state.clone(),
        x=x,
        weight=weight,
        bias=bias,
        groups=1,
        pad=KS - 1,
        conv=None,
        resets=resets,
    )

    # Test Triton implementation
    y_triton, state_triton = causal_conv1d_triton(
        conv_state=conv_state.clone(),
        x=x,
        weight=weight,
        bias=bias,
        groups=1,
        resets=resets,
    )

    # Compare outputs
    max_diff = (y_triton - y_pytorch).abs().max().item()
    mean_diff = (y_triton - y_pytorch).abs().mean().item()
    print(f"Forward pass - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    torch.testing.assert_close(
        y_triton,
        y_pytorch,
        rtol=2e-2,
        atol=2e-2,
        msg="Triton forward output should match PyTorch reference",
    )

    # Compare final states
    torch.testing.assert_close(
        state_triton,
        state_pytorch,
        rtol=1e-5,
        atol=1e-5,
        msg="Triton final state should match PyTorch reference",
    )


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
def test_causal_conv1d_triton_vs_pytorch_gradients():
    """Test that Triton and PyTorch implementations produce similar gradients."""
    torch.manual_seed(42)

    device = get_test_device()
    if not device.type == "cuda":
        pytest.skip("Triton kernel requires CUDA")

    dtype = torch.float32
    B = 2  # batch size
    T = 48  # sequence length
    F = 32  # feature dimension
    KS = 5  # kernel size

    # Create inputs for PyTorch version
    x_pytorch = torch.randn(B, T, F, device=device, dtype=dtype, requires_grad=True)
    weight_pytorch = torch.randn(F, F, KS, device=device, dtype=dtype, requires_grad=True)
    bias_pytorch = torch.randn(F, device=device, dtype=dtype, requires_grad=True)
    conv_state_pytorch = torch.randn(B, KS, F, device=device, dtype=dtype)

    # Create inputs for Triton version (clones with gradients)
    x_triton = x_pytorch.clone().detach().requires_grad_(True)
    weight_triton = weight_pytorch.clone().detach().requires_grad_(True)
    bias_triton = bias_pytorch.clone().detach().requires_grad_(True)
    conv_state_triton = conv_state_pytorch.clone().detach()

    # Create reset mask
    resets = torch.zeros(B, T, dtype=torch.bool, device=device)
    resets[0, 12] = True
    resets[1, 20] = True
    resets[0, 35] = True

    # Forward pass with PyTorch
    y_pytorch, _ = causal_conv1d_pytorch(
        conv_state=conv_state_pytorch,
        x=x_pytorch,
        weight=weight_pytorch,
        bias=bias_pytorch,
        groups=1,
        pad=KS - 1,
        conv=None,
        resets=resets,
    )

    # Forward pass with Triton
    y_triton, _ = causal_conv1d_triton(
        conv_state=conv_state_triton,
        x=x_triton,
        weight=weight_triton,
        bias=bias_triton,
        groups=1,
        resets=resets,
    )

    # Backward pass
    loss_pytorch = y_pytorch.mean()
    loss_pytorch.backward()

    loss_triton = y_triton.mean()
    loss_triton.backward()

    # Check that gradients exist
    assert x_pytorch.grad is not None, "PyTorch: x should have gradients"
    assert weight_pytorch.grad is not None, "PyTorch: weight should have gradients"
    assert bias_pytorch.grad is not None, "PyTorch: bias should have gradients"

    assert x_triton.grad is not None, "Triton: x should have gradients"
    assert weight_triton.grad is not None, "Triton: weight should have gradients"
    assert bias_triton.grad is not None, "Triton: bias should have gradients"

    # Check that gradients are non-zero
    assert not torch.all(x_pytorch.grad == 0), "PyTorch: x gradients should be non-zero"
    assert not torch.all(weight_pytorch.grad == 0), "PyTorch: weight gradients should be non-zero"
    assert not torch.all(bias_pytorch.grad == 0), "PyTorch: bias gradients should be non-zero"

    assert not torch.all(x_triton.grad == 0), "Triton: x gradients should be non-zero"
    assert not torch.all(weight_triton.grad == 0), "Triton: weight gradients should be non-zero"
    assert not torch.all(bias_triton.grad == 0), "Triton: bias gradients should be non-zero"

    # Compare gradients
    x_grad_diff = (x_triton.grad - x_pytorch.grad).abs().max().item()
    weight_grad_diff = (weight_triton.grad - weight_pytorch.grad).abs().max().item()
    bias_grad_diff = (bias_triton.grad - bias_pytorch.grad).abs().max().item()

    print(f"Gradient differences - x: {x_grad_diff:.6f}, weight: {weight_grad_diff:.6f}, bias: {bias_grad_diff:.6f}")

    # Compare gradients with reasonable tolerance
    torch.testing.assert_close(
        x_triton.grad,
        x_pytorch.grad,
        rtol=2e-2,
        atol=2e-2,
        msg="Input gradients should be similar between Triton and PyTorch",
    )

    torch.testing.assert_close(
        weight_triton.grad,
        weight_pytorch.grad,
        rtol=2e-2,
        atol=2e-2,
        msg="Weight gradients should be similar between Triton and PyTorch",
    )

    torch.testing.assert_close(
        bias_triton.grad,
        bias_pytorch.grad,
        rtol=2e-2,
        atol=2e-2,
        msg="Bias gradients should be similar between Triton and PyTorch",
    )


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
def test_causal_conv1d_triton_error_conditions():
    """Test that Triton implementation properly validates inputs."""
    device = get_test_device()
    if not device.type == "cuda":
        pytest.skip("Triton kernel requires CUDA")

    B, T, F, KS = 2, 32, 16, 4

    x = torch.randn(B, T, F, device=device)
    weight = torch.randn(F, F, KS, device=device)
    bias = torch.randn(F, device=device)
    conv_state = torch.randn(B, KS, F, device=device)
    resets = torch.zeros(B, T, dtype=torch.bool, device=device)

    # Test 1: groups != 1 should raise ValueError
    with pytest.raises(ValueError, match="groups=1"):
        causal_conv1d_triton(
            conv_state=conv_state,
            x=x,
            weight=weight,
            bias=bias,
            groups=F,  # depthwise mode not supported
            resets=resets,
        )

    # Test 2: missing resets should raise ValueError
    with pytest.raises(ValueError, match="per-timestep resets"):
        causal_conv1d_triton(
            conv_state=conv_state,
            x=x,
            weight=weight,
            bias=bias,
            groups=1,
            resets=None,
        )

    # Test 3: 1D resets should raise ValueError
    with pytest.raises(ValueError, match="per-timestep resets"):
        causal_conv1d_triton(
            conv_state=conv_state,
            x=x,
            weight=weight,
            bias=bias,
            groups=1,
            resets=resets[:, 0],  # 1D instead of 2D
        )


def test_causal_conv1d_pytorch_no_resets():
    """Test PyTorch implementation without per-timestep resets (fast path)."""
    torch.manual_seed(42)

    device = get_test_device()
    dtype = torch.float32
    B = 2
    T = 32
    F = 16
    KS = 4

    # Create a Conv1d module for the fast path
    conv = torch.nn.Conv1d(
        in_channels=F,
        out_channels=F,
        kernel_size=KS,
        padding=KS - 1,
        groups=1,
        bias=True,
        device=device,
        dtype=dtype,
    )

    x = torch.randn(B, T, F, device=device, dtype=dtype)
    conv_state = torch.randn(B, KS, F, device=device, dtype=dtype)

    # Test without resets (fast vectorized path)
    y_fast, state_fast = causal_conv1d_pytorch(
        conv_state=conv_state.clone(),
        x=x,
        weight=conv.weight,
        bias=conv.bias,
        groups=1,
        pad=KS - 1,
        conv=conv,
        resets=None,
    )

    # Test with batch-level resets (should still use fast path)
    resets_batch = torch.zeros(B, dtype=torch.bool, device=device)
    y_batch_reset, state_batch_reset = causal_conv1d_pytorch(
        conv_state=conv_state.clone(),
        x=x,
        weight=conv.weight,
        bias=conv.bias,
        groups=1,
        pad=KS - 1,
        conv=conv,
        resets=resets_batch,
    )

    # Both should produce valid outputs
    assert y_fast.shape == (B, T, F)
    assert state_fast.shape == (B, KS, F)
    assert y_batch_reset.shape == (B, T, F)
    assert state_batch_reset.shape == (B, KS, F)


if __name__ == "__main__":
    # Run tests manually
    print("Testing causal conv1d kernel implementations...")

    if TRITON_AVAILABLE and torch.cuda.is_available():
        print("\n=== Testing Triton forward pass ===")
        test_causal_conv1d_triton_vs_pytorch_forward()
        print("✓ Forward pass test passed")

        print("\n=== Testing Triton gradients ===")
        test_causal_conv1d_triton_vs_pytorch_gradients()
        print("✓ Gradient test passed")

        print("\n=== Testing Triton error conditions ===")
        test_causal_conv1d_triton_error_conditions()
        print("✓ Error condition test passed")
    else:
        print("Skipping Triton tests (Triton not available or no CUDA)")

    print("\n=== Testing PyTorch no resets ===")
    test_causal_conv1d_pytorch_no_resets()
    print("✓ PyTorch no resets test passed")

    print("\n✅ All tests passed!")
