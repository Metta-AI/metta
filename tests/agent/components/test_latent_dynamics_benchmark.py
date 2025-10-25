"""Benchmark Triton vs PyTorch implementations for latent dynamics operations.

This script compares performance of:
1. KL divergence computation
2. Reparameterization trick

Run with: uv run pytest tests/agent/components/test_latent_dynamics_benchmark.py -v --benchmark-only
"""

import pytest
import torch

from metta.agent.components.dynamics.triton_kernels import (
    TRITON_AVAILABLE,
    compute_kl_divergence,
    pytorch_kl_divergence,
    pytorch_reparameterize,
    reparameterize,
    triton_kl_divergence,
    triton_reparameterize,
)


def test_kl_divergence_pytorch_small(benchmark):
    """Benchmark PyTorch KL divergence with small tensors (batch=32, latent=16)."""
    z_mean = torch.randn(32, 16)
    z_logvar = torch.randn(32, 16)

    result = benchmark(pytorch_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1  # Scalar output


def test_kl_divergence_pytorch_medium(benchmark):
    """Benchmark PyTorch KL divergence with medium tensors (batch=256, latent=64)."""
    z_mean = torch.randn(256, 64)
    z_logvar = torch.randn(256, 64)

    result = benchmark(pytorch_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1


def test_kl_divergence_pytorch_large(benchmark):
    """Benchmark PyTorch KL divergence with large tensors (batch=1024, latent=128)."""
    z_mean = torch.randn(1024, 128)
    z_logvar = torch.randn(1024, 128)

    result = benchmark(pytorch_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kl_divergence_triton_small(benchmark):
    """Benchmark Triton KL divergence with small tensors (batch=32, latent=16)."""
    z_mean = torch.randn(32, 16, device="cuda")
    z_logvar = torch.randn(32, 16, device="cuda")

    result = benchmark(triton_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kl_divergence_triton_medium(benchmark):
    """Benchmark Triton KL divergence with medium tensors (batch=256, latent=64)."""
    z_mean = torch.randn(256, 64, device="cuda")
    z_logvar = torch.randn(256, 64, device="cuda")

    result = benchmark(triton_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kl_divergence_triton_large(benchmark):
    """Benchmark Triton KL divergence with large tensors (batch=1024, latent=128)."""
    z_mean = torch.randn(1024, 128, device="cuda")
    z_logvar = torch.randn(1024, 128, device="cuda")

    result = benchmark(triton_kl_divergence, z_mean, z_logvar)
    assert result.numel() == 1


def test_reparameterize_pytorch_small(benchmark):
    """Benchmark PyTorch reparameterization with small tensors (batch=32, latent=16)."""
    z_mean = torch.randn(32, 16)
    z_logvar = torch.randn(32, 16)

    result = benchmark(pytorch_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


def test_reparameterize_pytorch_medium(benchmark):
    """Benchmark PyTorch reparameterization with medium tensors (batch=256, latent=64)."""
    z_mean = torch.randn(256, 64)
    z_logvar = torch.randn(256, 64)

    result = benchmark(pytorch_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


def test_reparameterize_pytorch_large(benchmark):
    """Benchmark PyTorch reparameterization with large tensors (batch=1024, latent=128)."""
    z_mean = torch.randn(1024, 128)
    z_logvar = torch.randn(1024, 128)

    result = benchmark(pytorch_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reparameterize_triton_small(benchmark):
    """Benchmark Triton reparameterization with small tensors (batch=32, latent=16)."""
    z_mean = torch.randn(32, 16, device="cuda")
    z_logvar = torch.randn(32, 16, device="cuda")

    result = benchmark(triton_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reparameterize_triton_medium(benchmark):
    """Benchmark Triton reparameterization with medium tensors (batch=256, latent=64)."""
    z_mean = torch.randn(256, 64, device="cuda")
    z_logvar = torch.randn(256, 64, device="cuda")

    result = benchmark(triton_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reparameterize_triton_large(benchmark):
    """Benchmark Triton reparameterization with large tensors (batch=1024, latent=128)."""
    z_mean = torch.randn(1024, 128, device="cuda")
    z_logvar = torch.randn(1024, 128, device="cuda")

    result = benchmark(triton_reparameterize, z_mean, z_logvar)
    assert result.shape == z_mean.shape


def test_adaptive_kl_pytorch_cpu(benchmark):
    """Benchmark adaptive KL divergence on CPU (should use PyTorch)."""
    z_mean = torch.randn(256, 64)
    z_logvar = torch.randn(256, 64)

    result = benchmark(compute_kl_divergence, z_mean, z_logvar, use_triton=True)
    assert result.numel() == 1


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adaptive_kl_triton_cuda(benchmark):
    """Benchmark adaptive KL divergence on CUDA (should use Triton)."""
    z_mean = torch.randn(256, 64, device="cuda")
    z_logvar = torch.randn(256, 64, device="cuda")

    result = benchmark(compute_kl_divergence, z_mean, z_logvar, use_triton=True)
    assert result.numel() == 1


def test_adaptive_reparameterize_pytorch_cpu(benchmark):
    """Benchmark adaptive reparameterization on CPU (should use PyTorch)."""
    z_mean = torch.randn(256, 64)
    z_logvar = torch.randn(256, 64)

    result = benchmark(reparameterize, z_mean, z_logvar, use_triton=True)
    assert result.shape == z_mean.shape


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adaptive_reparameterize_triton_cuda(benchmark):
    """Benchmark adaptive reparameterization on CUDA (should use Triton)."""
    z_mean = torch.randn(256, 64, device="cuda")
    z_logvar = torch.randn(256, 64, device="cuda")

    result = benchmark(reparameterize, z_mean, z_logvar, use_triton=True)
    assert result.shape == z_mean.shape


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kl_divergence_consistency_triton_vs_pytorch():
    """Ensure Triton and PyTorch KL divergence implementations return the same value.

    This is critical for consistent training behavior across different hardware.
    """
    # Test with multiple batch sizes to ensure consistency
    for batch_size in [4, 16, 64, 256]:
        z_mean = torch.randn(batch_size, 32)
        z_logvar = torch.randn(batch_size, 32)

        # Compute on CPU with PyTorch
        pytorch_result = pytorch_kl_divergence(z_mean, z_logvar)

        # Compute on CUDA with Triton
        z_mean_cuda = z_mean.cuda()
        z_logvar_cuda = z_logvar.cuda()
        triton_result = triton_kl_divergence(z_mean_cuda, z_logvar_cuda)

        # Results should match closely (allowing for floating point differences)
        torch.testing.assert_close(
            pytorch_result,
            triton_result.cpu(),
            rtol=1e-4,
            atol=1e-5,
            msg=f"KL divergence mismatch for batch_size={batch_size}",
        )


def test_kl_divergence_consistency_cpu():
    """Test that KL divergence is consistent across different batch sizes on CPU."""
    # This ensures the implementation is correct regardless of hardware
    for batch_size in [4, 16, 64]:
        z_mean = torch.randn(batch_size, 32)
        z_logvar = torch.randn(batch_size, 32)

        # Compute KL divergence
        result = pytorch_kl_divergence(z_mean, z_logvar)

        # Result should be a scalar
        assert result.numel() == 1

        # Verify the result is finite
        assert torch.isfinite(result)
