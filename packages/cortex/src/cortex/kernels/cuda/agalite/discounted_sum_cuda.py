from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

_mod_path = os.path.dirname(__file__)
_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    sources = [
        os.path.join(_mod_path, "discounted_sum_binding.cpp"),
        os.path.join(_mod_path, "discounted_sum_kernels.cu"),
    ]
    _ext = load(
        name="agalite_discounted_sum",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        build_directory=None,
        verbose=False,
    )
    return _ext


def _prepare_start_state(start_state: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    broadcast_shape = x.shape[1:]
    aligned = start_state
    for _ in range(len(broadcast_shape) - start_state.dim()):
        aligned = aligned.unsqueeze(-1)
    if tuple(aligned.shape) != tuple(broadcast_shape):
        aligned = aligned.expand(broadcast_shape)
    return aligned.contiguous(), aligned.shape


class _DiscountedSumCUDA(Function):
    @staticmethod
    def forward(ctx, start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError("AGaLiTe CUDA extension unavailable")

        x_contig = x.contiguous()
        discounts_contig = discounts.contiguous()

        start_aligned, broadcast_shape = _prepare_start_state(start_state, x_contig)
        start_flat = start_aligned.reshape(-1).contiguous()

        T = x_contig.shape[0]
        feature_size = start_flat.numel()
        x_flat = x_contig.view(T, feature_size)
        discounts_flat = discounts_contig.view(T, feature_size)

        output_flat = ext.discounted_sum_forward(start_flat, x_flat, discounts_flat)

        ctx.save_for_backward(start_flat, discounts_flat, output_flat)
        ctx.start_shape = start_state.shape
        ctx.broadcast_shape = broadcast_shape
        ctx.x_shape = x_contig.shape

        return output_flat.view_as(x_contig)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError("AGaLiTe CUDA extension unavailable")

        start_flat, discounts_flat, output_flat = ctx.saved_tensors
        grad_flat = grad_output.contiguous().view(ctx.x_shape[0], -1)

        grad_start_flat, grad_x_flat, grad_discounts_flat = ext.discounted_sum_backward(
            grad_flat, discounts_flat, output_flat, start_flat
        )

        grad_start_broadcast = grad_start_flat.view(ctx.broadcast_shape)
        grad_start = torch.ops.aten.sum_to_size(grad_start_broadcast, ctx.start_shape)
        grad_x = grad_x_flat.view(ctx.x_shape)
        grad_discounts = grad_discounts_flat.view(ctx.x_shape)

        return grad_start, grad_x, grad_discounts


def discounted_sum_cuda(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Apply CUDA fused discounted sum if extension loads."""
    return _DiscountedSumCUDA.apply(start_state, x, discounts)


__all__ = ["discounted_sum_cuda"]
