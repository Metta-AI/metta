"""Advantage computation functions for Metta training."""

import importlib
from contextlib import nullcontext

import einops
import torch
from torch import Tensor

from metta.rl import mps

try:
    importlib.import_module("pufferlib._C")
except ImportError:
    raise ImportError("Failed to import C/CUDA kernel. Try: pip install --no-build-isolation") from None


def td_lambda_reverse_scan_pytorch(delta: Tensor, mask_next: Tensor, gamma_lambda: float) -> Tensor:
    running = torch.zeros_like(delta[:, -1])
    out = torch.zeros_like(delta)
    for t in range(delta.shape[1] - 1, -1, -1):
        running = delta[:, t] + gamma_lambda * mask_next[:, t] * running
        out[:, t] = running
    return out


def td_lambda_reverse_scan_cuda(delta: Tensor, mask_next: Tensor, gamma_lambda: float) -> Tensor:
    from cortex.kernels.cuda.agalite.discounted_sum_cuda import discounted_sum_cuda

    discounts = gamma_lambda * mask_next
    batch_size = delta.shape[0]
    start_state = torch.zeros((batch_size,), device=delta.device, dtype=delta.dtype)
    x_rev = delta.flip(1).transpose(0, 1)
    discounts_rev = discounts.flip(1).transpose(0, 1)
    out_rev = discounted_sum_cuda(start_state, x_rev, discounts_rev)
    return out_rev.transpose(0, 1).flip(1)


def td_lambda_reverse_scan(delta: Tensor, mask_next: Tensor, gamma_lambda: float) -> Tensor:
    from cortex.utils import select_backend

    fn = select_backend(
        triton_fn=None,
        pytorch_fn=td_lambda_reverse_scan_pytorch,
        tensor=delta,
        allow_triton=False,
        cuda_fn=td_lambda_reverse_scan_cuda,
        allow_cuda=True,
    )
    return fn(delta, mask_next, gamma_lambda)


def compute_delta_lambda(
    *,
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    _, tt = values.shape
    delta_lambda = torch.zeros_like(values)
    if tt <= 1:
        return delta_lambda

    terminal_next = dones[:, 1:]
    mask_next = 1.0 - terminal_next

    delta = rewards[:, 1:] + gamma * mask_next * values[:, 1:] - values[:, :-1]  # [B, TT-1]

    gamma_lambda = float(gamma * gae_lambda)
    delta_lambda[:, :-1] = td_lambda_reverse_scan(delta, mask_next, gamma_lambda)

    return delta_lambda


def compute_advantage(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    importance_sampling_ratio: Tensor,
    advantages: Tensor,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> Tensor:
    """CUDA kernel for puffer advantage with automatic CPU & MPS fallback."""

    # Move tensors to device and compute advantage
    # for mps (macbook pro)
    # for rocm (amd gpu) - pytorch has hip version
    if str(device) == "mps" or torch.version.hip is not None:
        return mps.advantage(
            values, rewards, dones, importance_sampling_ratio, vtrace_rho_clip, vtrace_c_clip, gamma, gae_lambda, device
        )

    # CUDA implementation using custom kernel
    tensors = [values, rewards, dones, importance_sampling_ratio, advantages]
    tensors = [t.to(device) for t in tensors]
    values, rewards, dones, importance_sampling_ratio, advantages = tensors

    # Create context manager that only applies CUDA device context if needed
    device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with device_context:
        torch.ops.pufferlib.compute_puff_advantage(
            values,
            rewards,
            dones,
            importance_sampling_ratio,
            advantages,
            gamma,
            gae_lambda,
            vtrace_rho_clip,
            vtrace_c_clip,
        )

    return advantages


def normalize_advantage_distributed(adv: Tensor, norm_adv: bool = True) -> Tensor:
    """Normalize advantages with distributed training support while preserving shape."""
    if not norm_adv:
        return adv

    if torch.distributed.is_initialized():
        # Compute local statistics
        adv_flat = adv.view(-1)
        local_sum = einops.rearrange(adv_flat.sum(), "-> 1")
        local_sq_sum = einops.rearrange((adv_flat * adv_flat).sum(), "-> 1")
        local_count = torch.tensor([adv_flat.numel()], dtype=adv.dtype, device=adv.device)

        # Combine statistics for single all_reduce
        stats = einops.rearrange([local_sum, local_sq_sum, local_count], "one float -> (float one)")
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        # Extract global statistics
        global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean * global_mean)
        global_std = torch.sqrt(global_var.clamp(min=1e-8))

        # Normalize and reshape back
        adv = (adv - global_mean) / (global_std + 1e-8)
    else:
        # Local normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv
