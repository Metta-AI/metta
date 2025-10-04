from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils import cpp_extension

_EXTENSION = None
_LOAD_ATTEMPTED = False


def _load_extension() -> Optional[object]:
    global _EXTENSION, _LOAD_ATTEMPTED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True

    src_dir = Path(__file__).resolve().parent
    sources = [str(src_dir / "agalite_kernels.cpp")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags: Optional[list[str]] = None

    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home and torch.cuda.is_available() and os.environ.get("METTA_FORCE_CPU_AGALITE_KERNEL", "0") != "1":
        sources.append(str(src_dir / "agalite_kernels.cu"))
        extra_cuda_cflags = ["-O3"]

    try:
        build_dir = src_dir / "_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXTENSION = cpp_extension.load(
            name="agalite_kernels",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=False,
        )
    except (OSError, RuntimeError) as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"Failed to build AGaLiTe fused kernels ({exc}). Falling back to TorchScript implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        _EXTENSION = None
    return _EXTENSION


def _prepare_start_state(start_state: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    broadcast_shape = x.shape[1:]
    aligned = start_state
    for _ in range(len(broadcast_shape) - start_state.dim()):
        aligned = aligned.unsqueeze(-1)
    if aligned.shape != broadcast_shape:
        aligned = aligned.expand(broadcast_shape)
    return aligned.contiguous(), aligned.shape


class _DiscountedSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("AGaLiTe fused kernels unavailable")

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
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("AGaLiTe fused kernels unavailable")

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


def fused_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Apply the fused discounted sum kernel.

    Raises:
        RuntimeError: If the native extension is unavailable.
    """

    return _DiscountedSumFunction.apply(start_state, x, discounts)


def extension_available() -> bool:
    """Return ``True`` if the fused kernels successfully loaded."""

    ext = _load_extension()
    return ext is not None


__all__ = ["fused_discounted_sum", "extension_available"]
