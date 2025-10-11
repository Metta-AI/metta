"""Triton-optimized kernels for latent dynamics model operations.

These kernels provide significant speedups for the KL divergence computation
and reparameterization trick, especially with larger batch sizes.
"""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.jit
    def kl_divergence_kernel(
        z_mean_ptr,
        z_logvar_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute KL divergence KL(q(z|x) || N(0,1)) element-wise.

        KL(q || p) = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        z_mean = tl.load(z_mean_ptr + offsets, mask=mask, other=0.0)
        z_logvar = tl.load(z_logvar_ptr + offsets, mask=mask, other=0.0)

        # Compute KL divergence: -0.5 * (1 + log_var - mean^2 - exp(log_var))
        kl = -0.5 * (1.0 + z_logvar - z_mean * z_mean - tl.exp(z_logvar))

        # Store result
        tl.store(output_ptr + offsets, kl, mask=mask)

    @triton.jit
    def reparameterize_kernel(
        z_mean_ptr,
        z_logvar_ptr,
        eps_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reparameterization trick: z = mu + sigma * epsilon.

        where sigma = exp(0.5 * log_var)
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        z_mean = tl.load(z_mean_ptr + offsets, mask=mask, other=0.0)
        z_logvar = tl.load(z_logvar_ptr + offsets, mask=mask, other=0.0)
        eps = tl.load(eps_ptr + offsets, mask=mask, other=0.0)

        # Compute z = mu + exp(0.5 * log_var) * eps
        std = tl.exp(0.5 * z_logvar)
        z = z_mean + std * eps

        # Store result
        tl.store(output_ptr + offsets, z, mask=mask)


def triton_kl_divergence(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence using Triton kernel.

    Args:
        z_mean: Mean of latent distribution (*, latent_dim)
        z_logvar: Log variance of latent distribution (*, latent_dim)

    Returns:
        kl_loss: KL divergence loss (scalar)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    # Flatten inputs
    z_mean_flat = z_mean.contiguous().view(-1)
    z_logvar_flat = z_logvar.contiguous().view(-1)
    n_elements = z_mean_flat.numel()

    # Allocate output
    output = torch.empty_like(z_mean_flat)

    # Launch kernel
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    kl_divergence_kernel[grid](
        z_mean_flat,
        z_logvar_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Sum and return
    return output.sum()


def triton_reparameterize(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick using Triton kernel.

    Args:
        z_mean: Mean of latent distribution (*, latent_dim)
        z_logvar: Log variance of latent distribution (*, latent_dim)

    Returns:
        z: Sampled latent variable (*, latent_dim)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    # Flatten inputs
    original_shape = z_mean.shape
    z_mean_flat = z_mean.contiguous().view(-1)
    z_logvar_flat = z_logvar.contiguous().view(-1)
    n_elements = z_mean_flat.numel()

    # Generate random noise
    eps = torch.randn_like(z_mean_flat)

    # Allocate output
    output = torch.empty_like(z_mean_flat)

    # Launch kernel
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    reparameterize_kernel[grid](
        z_mean_flat,
        z_logvar_flat,
        eps,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape and return
    return output.view(original_shape)


# Fallback implementations for when Triton is not available
def pytorch_kl_divergence(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for KL divergence computation."""
    kl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
    return kl.mean()


def pytorch_reparameterize(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for reparameterization trick."""
    std = torch.exp(0.5 * z_logvar)
    eps = torch.randn_like(std)
    return z_mean + eps * std


# Adaptive selection: use Triton if available, otherwise fall back to PyTorch
def compute_kl_divergence(z_mean: torch.Tensor, z_logvar: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """Compute KL divergence with automatic backend selection.

    Args:
        z_mean: Mean of latent distribution
        z_logvar: Log variance of latent distribution
        use_triton: Whether to use Triton if available (default: True)

    Returns:
        kl_loss: KL divergence loss
    """
    if use_triton and TRITON_AVAILABLE and z_mean.is_cuda:
        return triton_kl_divergence(z_mean, z_logvar)
    return pytorch_kl_divergence(z_mean, z_logvar)


def reparameterize(z_mean: torch.Tensor, z_logvar: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """Reparameterization trick with automatic backend selection.

    Args:
        z_mean: Mean of latent distribution
        z_logvar: Log variance of latent distribution
        use_triton: Whether to use Triton if available (default: True)

    Returns:
        z: Sampled latent variable
    """
    if use_triton and TRITON_AVAILABLE and z_mean.is_cuda:
        return triton_reparameterize(z_mean, z_logvar)
    return pytorch_reparameterize(z_mean, z_logvar)
