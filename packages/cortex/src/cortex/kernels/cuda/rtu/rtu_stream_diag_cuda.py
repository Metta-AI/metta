from __future__ import annotations

from typing import Optional

import torch
from torch.autograd import Function

from .rtu_seq_allin import backward_allin, forward_allin


def _act_to_id(name: str) -> int:
    n = name.lower()
    if n in ("silu", "swish"):
        return 0
    if n == "relu":
        return 1
    if n == "tanh":
        return 2
    if n in ("linear", "identity"):
        return 3
    raise ValueError(f"Unsupported activation: {name}")


class _RTUStreamDiagCUDASeqAllIn(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_btd: torch.Tensor,
        nu_log: torch.Tensor,
        theta_log: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation_name: str,
        hc1_init_bh: torch.Tensor,
        hc2_init_bh: torch.Tensor,
        trace_in: Optional[tuple[torch.Tensor, ...]] = None,
        resets_bt: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, T, H = x_btd.shape
        assert H == nu_log.numel(), "D must equal H for diagonal map"

        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=x_btd.device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=x_btd.device)

        if trace_in is None:
            zeros = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
            trace_in = (zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros)
        (
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_w1_c1_in,
            E_w1_c2_in,
            E_w2_c1_in,
            E_w2_c2_in,
        ) = trace_in

        act_id = _act_to_id(activation_name)

        (
            y_btd_2h,
            pre1_bth,
            pre2_bth,
            final_hc1_bh,
            final_hc2_bh,
            E_nu_c1_out,
            E_nu_c2_out,
            E_th_c1_out,
            E_th_c2_out,
            E_w1_c1_out,
            E_w1_c2_out,
            E_w2_c1_out,
            E_w2_c2_out,
        ) = forward_allin(
            x_btd.contiguous(),
            nu_log.contiguous(),
            theta_log.contiguous(),
            w1.contiguous(),
            w2.contiguous(),
            hc1_init_bh.contiguous(),
            hc2_init_bh.contiguous(),
            E_nu_c1_in.contiguous(),
            E_nu_c2_in.contiguous(),
            E_th_c1_in.contiguous(),
            E_th_c2_in.contiguous(),
            E_w1_c1_in.contiguous(),
            E_w1_c2_in.contiguous(),
            E_w2_c1_in.contiguous(),
            E_w2_c2_in.contiguous(),
            resets_bt.to(torch.uint8).contiguous(),
            act_id,
        )

        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
            pre1_bth,
            pre2_bth,
            hc1_init_bh,
            hc2_init_bh,
            resets_bt.to(torch.uint8),
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_w1_c1_in,
            E_w1_c2_in,
            E_w2_c1_in,
            E_w2_c2_in,
        )
        ctx.act_id = act_id

        trace_out = (
            E_nu_c1_out,
            E_nu_c2_out,
            E_th_c1_out,
            E_th_c2_out,
            E_w1_c1_out,
            E_w1_c2_out,
            E_w2_c1_out,
            E_w2_c2_out,
        )
        return y_btd_2h, final_hc1_bh, final_hc2_bh, trace_out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y_btd_2h: torch.Tensor,
        grad_final_hc1: torch.Tensor,
        grad_final_hc2: torch.Tensor,
        grad_trace_out,
    ):
        (
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
            pre1_bth,
            pre2_bth,
            hc1_init_bh,
            hc2_init_bh,
            resets_u8,
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_w1_c1_in,
            E_w1_c2_in,
            E_w2_c1_in,
            E_w2_c2_in,
        ) = ctx.saved_tensors
        act_id = ctx.act_id

        (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_w1_h,
            grad_w2_h,
            grad_hc1_init,
            grad_hc2_init,
        ) = backward_allin(
            grad_y_btd_2h.contiguous(),
            x_btd.contiguous(),
            nu_log.contiguous(),
            theta_log.contiguous(),
            w1.contiguous(),
            w2.contiguous(),
            pre1_bth.contiguous(),
            pre2_bth.contiguous(),
            hc1_init_bh.contiguous(),
            hc2_init_bh.contiguous(),
            resets_u8.contiguous(),
            E_nu_c1_in.contiguous(),
            E_nu_c2_in.contiguous(),
            E_th_c1_in.contiguous(),
            E_th_c2_in.contiguous(),
            E_w1_c1_in.contiguous(),
            E_w1_c2_in.contiguous(),
            E_w2_c1_in.contiguous(),
            E_w2_c2_in.contiguous(),
            act_id,
        )

        return (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_w1_h,
            grad_w2_h,
            None,
            grad_hc1_init,
            grad_hc2_init,
            None,
            None,
        )


def rtu_stream_diag_cuda(
    *,
    x_btd: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_name: str,
    hc1_init_bh: torch.Tensor,
    hc2_init_bh: torch.Tensor,
    trace_in: Optional[tuple[torch.Tensor, ...]] = None,
    resets_bt: Optional[torch.Tensor] = None,
):
    """CUDA diagonal RTU streaming kernel.

    Computes streaming RTU with diagonal input weights using fused CUDA kernels.
    """
    y, h1, h2, trace = _RTUStreamDiagCUDASeqAllIn.apply(
        x_btd,
        nu_log,
        theta_log,
        w1,
        w2,
        activation_name,
        hc1_init_bh,
        hc2_init_bh,
        trace_in,
        resets_bt,
    )
    return y, (h1, h2), trace


__all__ = ["rtu_stream_diag_cuda"]
