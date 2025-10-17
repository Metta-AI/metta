"""Streaming (chunk-wise) PyTorch kernel for low-rank RTUs.

This variant supports processing long sequences in chunks while allowing
detached hidden states between chunks without losing cross-boundary credit
assignment. It does so by carrying compact forward-mode traces ("eligibility"
traces) across subsequences, and adding a single boundary correction in the
autograd backward using the chunk-head adjoint.

Public API (functional): ``rtu_sequence_pytorch_streaming``
Returns outputs, final state, and ``trace_out``. For streaming, call this per
chunk, detach ``(hc1,hc2)`` and the returned ``trace_out``, and feed both back
as inputs for the next chunk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd import Function

# Reuse the activation from the baseline kernel to keep parity
from cortex.kernels.pytorch.rtu import _act_and_deriv


# ---- Packing helpers ----
def _zeros_like_traces(B: int, T: int, D: int, H: int, R: int, *, device, dtype) -> tuple[torch.Tensor, ...]:
    # Diagonal (nu/theta) traces (B,H)
    E_nu_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_nu_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    # U-base traces (B,D,H)
    A_U1_c1 = torch.zeros(B, D, H, device=device, dtype=dtype)
    A_U1_c2 = torch.zeros(B, D, H, device=device, dtype=dtype)
    A_U2_c1 = torch.zeros(B, D, H, device=device, dtype=dtype)
    A_U2_c2 = torch.zeros(B, D, H, device=device, dtype=dtype)
    # V traces (B,R,H)
    C_V1_c1 = torch.zeros(B, R, H, device=device, dtype=dtype)
    C_V1_c2 = torch.zeros(B, R, H, device=device, dtype=dtype)
    C_V2_c1 = torch.zeros(B, R, H, device=device, dtype=dtype)
    C_V2_c2 = torch.zeros(B, R, H, device=device, dtype=dtype)
    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        A_U1_c1,
        A_U1_c2,
        A_U2_c1,
        A_U2_c2,
        C_V1_c1,
        C_V1_c2,
        C_V2_c1,
        C_V2_c2,
    )


def _unpack_traces(trace: tuple[torch.Tensor, ...]):
    (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        A_U1_c1,
        A_U1_c2,
        A_U2_c1,
        A_U2_c2,
        C_V1_c1,
        C_V1_c2,
        C_V2_c1,
        C_V2_c2,
    ) = trace
    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        A_U1_c1,
        A_U1_c2,
        A_U2_c1,
        A_U2_c2,
        C_V1_c1,
        C_V1_c2,
        C_V2_c1,
        C_V2_c2,
    )


class _LinearRTUFunctionLR_Streaming(Function):
    @staticmethod
    def forward(
        ctx,
        x_btd: torch.Tensor,  # (B,T,D)
        nu_log: torch.Tensor,  # (H,)
        theta_log: torch.Tensor,  # (H,)
        U1: torch.Tensor,  # (D,R)
        U2: torch.Tensor,  # (D,R)
        V1: torch.Tensor,  # (R,H)
        V2: torch.Tensor,  # (R,H)
        activation_name: str,
        hc1_init_bh: torch.Tensor,  # (B,H)
        hc2_init_bh: torch.Tensor,  # (B,H)
        trace_in: Optional[tuple[torch.Tensor, ...]] = None,
        resets_bt: Optional[torch.Tensor] = None,  # (B,T) bool or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, T, D = x_btd.shape
        H = nu_log.shape[0]
        R = U1.shape[1]

        device = x_btd.device
        dtype = x_btd.dtype

        # Decode diagonal dynamics
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        theta = torch.exp(theta_log)  # (H,)
        g = r * torch.cos(theta)
        phi = r * torch.sin(theta)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        pre1_bth = x_btd.new_empty(B, T, H)
        pre2_bth = x_btd.new_empty(B, T, H)

        hc1 = hc1_init_bh
        hc2 = hc2_init_bh

        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=device)

        # Projections
        a1_btr = torch.einsum("btd,dr->btr", x_btd, U1)  # (B,T,R)
        a2_btr = torch.einsum("btd,dr->btr", x_btd, U2)
        u1_bth = torch.einsum("btr,rh->bth", a1_btr, V1)  # (B,T,H)
        u2_bth = torch.einsum("btr,rh->bth", a2_btr, V2)

        # Initialize carried traces
        if trace_in is None:
            (
                E_nu_c1,
                E_nu_c2,
                E_th_c1,
                E_th_c2,
                A_U1_c1,
                A_U1_c2,
                A_U2_c1,
                A_U2_c2,
                C_V1_c1,
                C_V1_c2,
                C_V2_c1,
                C_V2_c2,
            ) = _zeros_like_traces(B, T, D, H, R, device=device, dtype=dtype)
        else:
            (
                E_nu_c1,
                E_nu_c2,
                E_th_c1,
                E_th_c2,
                A_U1_c1,
                A_U1_c2,
                A_U2_c1,
                A_U2_c2,
                C_V1_c1,
                C_V1_c2,
                C_V2_c1,
                C_V2_c2,
            ) = _unpack_traces(trace_in)

        # Precompute derivatives wrt (nu_log, theta_log) for diag traces
        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        d_g_d_nu = -exp_nu_log * g
        d_phi_d_nu = -exp_nu_log * phi
        d_gamma_d_nu = exp_nu_log * r_val * r_val / sqrt_1_minus_r2
        d_g_d_th = -phi * exp_th_log
        d_phi_d_th = g * exp_th_log

        g_1h = g.view(1, 1, H)
        phi_1h = phi.view(1, 1, H)
        gamma_1h = gamma.view(1, 1, H)

        d_g_d_nu_bh = d_g_d_nu.view(1, H)
        d_phi_d_nu_bh = d_phi_d_nu.view(1, H)
        d_gamma_d_nu_bh = d_gamma_d_nu.view(1, H)
        d_g_d_th_bh = d_g_d_th.view(1, H)
        d_phi_d_th_bh = d_phi_d_th.view(1, H)

        for t in range(T):
            if resets_bt.any():
                m_b11 = resets_bt[:, t].view(B, 1, 1).to(dtype=dtype)
                one_minus_b11 = 1.0 - m_b11
                # Zero traces and states on resets (row-wise)
                A_U1_c1 = A_U1_c1 * one_minus_b11
                A_U1_c2 = A_U1_c2 * one_minus_b11
                A_U2_c1 = A_U2_c1 * one_minus_b11
                A_U2_c2 = A_U2_c2 * one_minus_b11
                C_V1_c1 = C_V1_c1 * one_minus_b11
                C_V1_c2 = C_V1_c2 * one_minus_b11
                C_V2_c1 = C_V2_c1 * one_minus_b11
                C_V2_c2 = C_V2_c2 * one_minus_b11

                m_b1 = resets_bt[:, t].view(B, 1).to(dtype=dtype)
                one_minus_b1 = 1.0 - m_b1
                hc1 = hc1 * one_minus_b1
                hc2 = hc2 * one_minus_b1
                E_nu_c1 = E_nu_c1 * one_minus_b1
                E_nu_c2 = E_nu_c2 * one_minus_b1
                E_th_c1 = E_th_c1 * one_minus_b1
                E_th_c2 = E_th_c2 * one_minus_b1

            u1_t = u1_bth[:, t, :]
            u2_t = u2_bth[:, t, :]

            # Linear preactivations
            c1_t = g * hc1 - phi * hc2 + gamma * u1_t
            c2_t = g * hc2 + phi * hc1 + gamma * u2_t
            pre1_bth[:, t, :] = c1_t
            pre2_bth[:, t, :] = c2_t
            hc1, hc2 = c1_t, c2_t

            # Update carried traces (base/compact form)
            x_t = x_btd[:, t, :]  # (B,D)
            a1_t = a1_btr[:, t, :]  # (B,R)
            a2_t = a2_btr[:, t, :]

            Au11, Au12 = A_U1_c1, A_U1_c2
            Au21, Au22 = A_U2_c1, A_U2_c2
            Cv11, Cv12 = C_V1_c1, C_V1_c2
            Cv21, Cv22 = C_V2_c1, C_V2_c2

            # U1 base traces (B,D,H)
            X_bdh = x_t.unsqueeze(-1)  # (B,D,1)
            A_U1_c1 = g_1h * Au11 - phi_1h * Au12 + gamma_1h * X_bdh
            A_U1_c2 = g_1h * Au12 + phi_1h * Au11

            # U2 base traces (B,D,H)
            A_U2_c2 = g_1h * Au22 + phi_1h * Au21 + gamma_1h * X_bdh
            A_U2_c1 = g_1h * Au21 - phi_1h * Au22

            # V1 traces (B,R,H)
            A1_br1 = a1_t.unsqueeze(-1)  # (B,R,1)
            C_V1_c1 = g_1h * Cv11 - phi_1h * Cv12 + gamma_1h * A1_br1
            C_V1_c2 = g_1h * Cv12 + phi_1h * Cv11

            # V2 traces (B,R,H)
            A2_br1 = a2_t.unsqueeze(-1)
            C_V2_c2 = g_1h * Cv22 + phi_1h * Cv21 + gamma_1h * A2_br1
            C_V2_c1 = g_1h * Cv21 - phi_1h * Cv22

            # Diagonal parameter traces (B,H)
            # Using previous state before update is correct per RTRL recurrence
            hc1_prev = pre1_bth[:, t - 1, :] if t > 0 else hc1_init_bh
            hc2_prev = pre2_bth[:, t - 1, :] if t > 0 else hc2_init_bh

            Enu1_old, Enu2_old = E_nu_c1, E_nu_c2
            Eth1_old, Eth2_old = E_th_c1, E_th_c2

            E_nu_c1 = (
                d_g_d_nu_bh * hc1_prev
                + g.view(1, H) * Enu1_old
                - d_phi_d_nu_bh * hc2_prev
                - phi.view(1, H) * Enu2_old
                + d_gamma_d_nu_bh * u1_t
            )
            E_nu_c2 = (
                d_g_d_nu_bh * hc2_prev
                + g.view(1, H) * Enu2_old
                + d_phi_d_nu_bh * hc1_prev
                + phi.view(1, H) * Enu1_old
                + d_gamma_d_nu_bh * u2_t
            )
            E_th_c1 = (
                d_g_d_th_bh * hc1_prev + g.view(1, H) * Eth1_old - d_phi_d_th_bh * hc2_prev - phi.view(1, H) * Eth2_old
            )
            E_th_c2 = (
                d_g_d_th_bh * hc2_prev + g.view(1, H) * Eth2_old + d_phi_d_th_bh * hc1_prev + phi.view(1, H) * Eth1_old
            )

        # Activation
        y1_bth, _ = _act_and_deriv(pre1_bth, activation_name)
        y2_bth, _ = _act_and_deriv(pre2_bth, activation_name)
        y_btd_2h = torch.cat([y1_bth, y2_bth], dim=-1)

        trace_out = (
            E_nu_c1,
            E_nu_c2,
            E_th_c1,
            E_th_c2,
            A_U1_c1,
            A_U1_c2,
            A_U2_c1,
            A_U2_c2,
            C_V1_c1,
            C_V1_c2,
            C_V2_c1,
            C_V2_c2,
        )

        # Save for backward
        if trace_in is None:
            trace_in = _zeros_like_traces(B, T, D, H, R, device=device, dtype=dtype)
        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            pre1_bth,
            pre2_bth,
            g,
            phi,
            gamma,
            hc1_init_bh,
            hc2_init_bh,
            *trace_in,  # carried-in traces for boundary correction
            resets_bt,
        )
        ctx.activation_name = activation_name

        final_hc1_bh = pre1_bth[:, -1, :] if T > 0 else hc1_init_bh
        final_hc2_bh = pre2_bth[:, -1, :] if T > 0 else hc2_init_bh
        return y_btd_2h, final_hc1_bh, final_hc2_bh, trace_out

    @staticmethod
    def backward(
        ctx,
        grad_y_btd_2h: torch.Tensor,  # (B,T,2H)
        grad_final_hc1: torch.Tensor,  # unused
        grad_final_hc2: torch.Tensor,  # unused
        grad_trace_out: Optional[tuple[torch.Tensor, ...]],  # None; we don't backprop into traces
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        (
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            pre1_bth,
            pre2_bth,
            g,
            phi,
            gamma,
            hc1_init_bh,
            hc2_init_bh,
            E_nu_c1_in,
            E_nu_c2_in,
            E_th_c1_in,
            E_th_c2_in,
            A_U1_c1_in,
            A_U1_c2_in,
            A_U2_c1_in,
            A_U2_c2_in,
            C_V1_c1_in,
            C_V1_c2_in,
            C_V2_c1_in,
            C_V2_c2_in,
            resets_bt,
        ) = saved
        activation_name = ctx.activation_name

        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        # Local adjoints from activation
        _, d1 = _act_and_deriv(pre1_bth, activation_name)  # (B,T,H)
        _, d2 = _act_and_deriv(pre2_bth, activation_name)
        gy1 = grad_y_btd_2h[:, :, :H]
        gy2 = grad_y_btd_2h[:, :, H:]
        eta1 = d1 * gy1
        eta2 = d2 * gy2

        # Precompute direct drives for diag/gamma contributions
        a1_btr = torch.einsum("btd,dr->btr", x_btd, U1)
        a2_btr = torch.einsum("btd,dr->btr", x_btd, U2)
        u1_bth = torch.einsum("btr,rh->bth", a1_btr, V1)
        u2_bth = torch.einsum("btr,rh->bth", a2_btr, V2)

        # Reverse-time suffix scan for adjoints
        grad_x_btd = torch.zeros_like(x_btd)
        GW1_DH = torch.zeros(D, H, device=x_btd.device, dtype=x_btd.dtype)
        GW2_DH = torch.zeros(D, H, device=x_btd.device, dtype=x_btd.dtype)
        # Accumulators for diag gradients in reverse-mode form
        dg_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
        dphi_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
        dgamma_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)

        s1_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        s2_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)

        g_bh = g.view(1, H)
        phi_bh = phi.view(1, H)
        gamma_bh = gamma.view(1, H)

        for t in range(T - 1, -1, -1):
            d1_t = eta1[:, t, :]
            d2_t = eta2[:, t, :]
            if resets_bt.any():
                m = resets_bt[:, t].view(B, 1).to(dtype=d1_t.dtype)
                one_minus = 1.0 - m
                s1 = d1_t + one_minus * (g_bh * s1_next + phi_bh * s2_next)
                s2 = d2_t + one_minus * (g_bh * s2_next - phi_bh * s1_next)
            else:
                s1 = d1_t + g_bh * s1_next + phi_bh * s2_next
                s2 = d2_t + g_bh * s2_next - phi_bh * s1_next

            # Reverse-mode contributions to low-rank and input grads
            s1g = s1 * gamma_bh
            s2g = s2 * gamma_bh

            X_t = x_btd[:, t, :]
            GW1_DH += X_t.t() @ s1g
            GW2_DH += X_t.t() @ s2g

            tmp1_BR = s1g @ V1.t()
            tmp2_BR = s2g @ V2.t()
            grad_x_btd[:, t, :] += tmp1_BR @ U1.t() + tmp2_BR @ U2.t()

            # Reverse-mode contributions to diag grads using c_prev and u_t
            if t == 0:
                cprev1 = hc1_init_bh
                cprev2 = hc2_init_bh
            else:
                cprev1 = pre1_bth[:, t - 1, :]
                cprev2 = pre2_bth[:, t - 1, :]
            if resets_bt.any():
                m = resets_bt[:, t].view(B, 1).to(dtype=cprev1.dtype)
                one_minus = 1.0 - m
                cprev1 = cprev1 * one_minus
                cprev2 = cprev2 * one_minus

            dg_sum += torch.sum(s1 * cprev1 + s2 * cprev2, dim=0)
            dphi_sum += torch.sum(-s1 * cprev2 + s2 * cprev1, dim=0)
            dgamma_sum += torch.sum(s1 * u1_bth[:, t, :] + s2 * u2_bth[:, t, :], dim=0)

            s1_next, s2_next = s1, s2

        # Boundary adjoint: need λ at c_{t0-1} to pair with E_{t0-1}
        lambda0_c1 = s1_next  # λ at c_{t0}
        lambda0_c2 = s2_next
        lam_prev_c1 = g.view(1, H) * lambda0_c1 + phi.view(1, H) * lambda0_c2
        lam_prev_c2 = -phi.view(1, H) * lambda0_c1 + g.view(1, H) * lambda0_c2
        if resets_bt.any():
            head_mask = 1.0 - resets_bt[:, 0].view(B, 1).to(dtype=lambda0_c1.dtype)
            lam_prev_c1 = lam_prev_c1 * head_mask
            lam_prev_c2 = lam_prev_c2 * head_mask

        # Low-rank parameter grads (local)
        grad_U1_DR = GW1_DH @ V1.t()
        grad_U2_DR = GW2_DH @ V2.t()
        grad_V1_RH = U1.t() @ GW1_DH
        grad_V2_RH = U2.t() @ GW2_DH
        # Diagonal parameter grads via reverse-mode accumulators
        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        gamma_ = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        grad_nu_log_h = -exp_nu_log * (dg_sum * g + dphi_sum * phi) + exp_nu_log * (r_val * r_val / gamma_) * dgamma_sum
        grad_theta_log_h = exp_th_log * (-dg_sum * phi + dphi_sum * g)

        # ---- Boundary corrections using carried-in traces ----
        # U1 correction (B,D,H) -> (D,H) -> (D,R)
        S_U1_DH = torch.sum(
            A_U1_c1_in * lam_prev_c1.unsqueeze(1) + A_U1_c2_in * lam_prev_c2.unsqueeze(1),
            dim=0,
        )
        grad_U1_DR = grad_U1_DR + S_U1_DH @ V1.t()

        # U2
        S_U2_DH = torch.sum(
            A_U2_c1_in * lam_prev_c1.unsqueeze(1) + A_U2_c2_in * lam_prev_c2.unsqueeze(1),
            dim=0,
        )
        grad_U2_DR = grad_U2_DR + S_U2_DH @ V2.t()

        # V1, V2 corrections
        grad_V1_RH = grad_V1_RH + torch.sum(
            C_V1_c1_in * lam_prev_c1.unsqueeze(1) + C_V1_c2_in * lam_prev_c2.unsqueeze(1),
            dim=0,
        )
        grad_V2_RH = grad_V2_RH + torch.sum(
            C_V2_c1_in * lam_prev_c1.unsqueeze(1) + C_V2_c2_in * lam_prev_c2.unsqueeze(1),
            dim=0,
        )

        # Diagonal corrections
        grad_nu_log_h = grad_nu_log_h + torch.sum(lam_prev_c1 * E_nu_c1_in + lam_prev_c2 * E_nu_c2_in, dim=0)
        grad_theta_log_h = grad_theta_log_h + torch.sum(lam_prev_c1 * E_th_c1_in + lam_prev_c2 * E_th_c2_in, dim=0)

        # Initial-state grads (to enable gradient flow across chunks if not detached)
        grad_hc1_init = lambda0_c1
        grad_hc2_init = lambda0_c2

        return (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_U1_DR,
            grad_U2_DR,
            grad_V1_RH,
            grad_V2_RH,
            None,  # activation_name
            grad_hc1_init,
            grad_hc2_init,
            None,  # trace_in (no grad)
            None,  # resets
        )


def rtu_sequence_pytorch_streaming(
    *,
    x_btd: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    U1: torch.Tensor,
    U2: torch.Tensor,
    V1: torch.Tensor,
    V2: torch.Tensor,
    activation_name: str,
    hc1_init_bh: torch.Tensor,
    hc2_init_bh: torch.Tensor,
    trace_in: Optional[tuple[torch.Tensor, ...]] = None,
    resets_bt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Streaming functional RTU (PyTorch) that carries compact traces across chunks.

    Returns:
        y_btd_2h, (final_hc1_bh, final_hc2_bh), trace_out
    """
    y, h1, h2, trace_out = _LinearRTUFunctionLR_Streaming.apply(
        x_btd,
        nu_log,
        theta_log,
        U1,
        U2,
        V1,
        V2,
        activation_name,
        hc1_init_bh,
        hc2_init_bh,
        trace_in,
        resets_bt,
    )
    return y, (h1, h2), trace_out


__all__ = ["rtu_sequence_pytorch_streaming"]
