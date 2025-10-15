from __future__ import annotations

from typing import Optional

import torch
from torch.autograd import Function

from .rtu_seq_full import backward_full, forward_full


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


class _RTUStreamFullCUDASeq(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_btd: torch.Tensor,  # [B,T,D]
        nu_log: torch.Tensor,  # [H]
        theta_log: torch.Tensor,  # [H]
        Wc1: torch.Tensor,  # [D,H]
        Wc2: torch.Tensor,  # [D,H]
        activation_name: str,
        hc1_init_bh: torch.Tensor,  # [B,H]
        hc2_init_bh: torch.Tensor,  # [B,H]
        trace_in: Optional[tuple[torch.Tensor, ...]] = None,
        resets_bt: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, T, D = x_btd.shape
        H = nu_log.shape[0]
        assert Wc1.shape == (D, H) and Wc2.shape == (D, H)

        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=x_btd.device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=x_btd.device)

        if trace_in is None:
            zeros_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
            zeros_bdh = torch.zeros(B, D, H, device=x_btd.device, dtype=x_btd.dtype)
            trace_in = (zeros_bh, zeros_bh, zeros_bh, zeros_bh, zeros_bdh, zeros_bdh, zeros_bdh, zeros_bdh)

        (
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_W1_c1_in,
            E_W1_c2_in,
            E_W2_c1_in,
            E_W2_c2_in,
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
            E_W1_c1_out,
            E_W1_c2_out,
            E_W2_c1_out,
            E_W2_c2_out,
        ) = forward_full(
            x_btd.contiguous(),
            nu_log.contiguous(),
            theta_log.contiguous(),
            Wc1.contiguous(),
            Wc2.contiguous(),
            hc1_init_bh.contiguous(),
            hc2_init_bh.contiguous(),
            E_nu_c1_in.contiguous(),
            E_nu_c2_in.contiguous(),
            E_th_c1_in.contiguous(),
            E_th_c2_in.contiguous(),
            E_W1_c1_in.contiguous(),
            E_W1_c2_in.contiguous(),
            E_W2_c1_in.contiguous(),
            E_W2_c2_in.contiguous(),
            resets_bt.to(torch.uint8).contiguous(),
            act_id,
        )

        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            Wc1,
            Wc2,
            pre1_bth,
            pre2_bth,
            hc1_init_bh,
            hc2_init_bh,
            resets_bt.to(torch.uint8),
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_W1_c1_in,
            E_W1_c2_in,
            E_W2_c1_in,
            E_W2_c2_in,
        )
        ctx.act_id = act_id

        trace_out = (
            E_nu_c1_out,
            E_nu_c2_out,
            E_th_c1_out,
            E_th_c2_out,
            E_W1_c1_out,
            E_W1_c2_out,
            E_W2_c1_out,
            E_W2_c2_out,
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
            Wc1,
            Wc2,
            pre1_bth,
            pre2_bth,
            hc1_init_bh,
            hc2_init_bh,
            resets_u8,
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            E_W1_c1_in,
            E_W1_c2_in,
            E_W2_c1_in,
            E_W2_c2_in,
        ) = ctx.saved_tensors
        act_id = ctx.act_id

        (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_Wc1,
            grad_Wc2,
            grad_hc1_init,
            grad_hc2_init,
        ) = backward_full(
            grad_y_btd_2h.contiguous(),
            x_btd.contiguous(),
            nu_log.contiguous(),
            theta_log.contiguous(),
            Wc1.contiguous(),
            Wc2.contiguous(),
            pre1_bth.contiguous(),
            pre2_bth.contiguous(),
            hc1_init_bh.contiguous(),
            hc2_init_bh.contiguous(),
            resets_u8.contiguous(),
            E_nu_c1_in.contiguous(),
            E_nu_c2_in.contiguous(),
            E_th_c1_in.contiguous(),
            E_th_c2_in.contiguous(),
            E_W1_c1_in.contiguous(),
            E_W1_c2_in.contiguous(),
            E_W2_c1_in.contiguous(),
            E_W2_c2_in.contiguous(),
            act_id,
        )

        return (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_Wc1,
            grad_Wc2,
            None,
            grad_hc1_init,
            grad_hc2_init,
            None,
            None,
        )


def rtu_stream_full_cuda_seq_allin(
    *,
    x_btd: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    Wc1: torch.Tensor,
    Wc2: torch.Tensor,
    activation_name: str,
    hc1_init_bh: torch.Tensor,
    hc2_init_bh: torch.Tensor,
    trace_in: Optional[tuple[torch.Tensor, ...]] = None,
    resets_bt: Optional[torch.Tensor] = None,
):
    y, h1, h2, trace = _RTUStreamFullCUDASeq.apply(
        x_btd,
        nu_log,
        theta_log,
        Wc1,
        Wc2,
        activation_name,
        hc1_init_bh,
        hc2_init_bh,
        trace_in,
        resets_bt,
    )
    return y, (h1, h2), trace


__all__ = ["rtu_stream_full_cuda_seq_allin"]
