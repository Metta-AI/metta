from __future__ import annotations

import torch
from torch import Tensor

from cortex.utils import select_backend


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
    fn = select_backend(
        triton_fn=None,
        pytorch_fn=td_lambda_reverse_scan_pytorch,
        tensor=delta,
        allow_triton=False,
        cuda_fn=td_lambda_reverse_scan_cuda,
        allow_cuda=True,
    )
    return fn(delta, mask_next, gamma_lambda)


__all__ = ["td_lambda_reverse_scan", "td_lambda_reverse_scan_cuda", "td_lambda_reverse_scan_pytorch"]
