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
