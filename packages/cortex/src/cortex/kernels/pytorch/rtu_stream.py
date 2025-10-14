"""Streaming (chunk-wise) PyTorch kernels for RTUs.

This module provides a streaming variant that supports processing long
sequences in chunks while allowing detached hidden states between chunks
without losing cross-boundary credit assignment. It does so by carrying compact
forward-mode traces ("eligibility" traces) across subsequences, and adding a
single boundary correction in the autograd backward using the chunk-head
adjoint.

Diagonal input maps (lightweight): ``rtu_stream_diag_pytorch``.

Return values: outputs, final state, and ``trace_out``. For streaming, call per
chunk, detach ``(hc1,hc2)`` and the returned ``trace_out``, and feed both back
as inputs for the next chunk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd import Function


# ------------------------------------------------------------
# Diagonal input-map streaming RTU (new, lightweight variant)
# ------------------------------------------------------------

# ---- Activation + derivative (SiLU, ReLU, Tanh, Identity) ----
def _act_and_deriv(z: torch.Tensor, activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    name = activation.lower()
    if name in ["silu", "swish"]:
        y = torch.nn.functional.silu(z)
        s = torch.sigmoid(z)
        dy = s * (1.0 + z * (1.0 - s))  # d/dz [z*sigmoid(z)]
        return y, dy
    elif name == "relu":
        y = torch.relu(z)
        dy = (z > 0).to(z.dtype)
        return y, dy
    elif name == "tanh":
        y = torch.tanh(z)
        dy = 1.0 - y * y
        return y, dy
    elif name in ["linear", "identity"]:
        return z, torch.ones_like(z)
    else:
        raise ValueError(f"Unsupported activation for RTU: {activation}")


def _zeros_like_traces_diag(B: int, H: int, *, device, dtype) -> tuple[torch.Tensor, ...]:
    """Allocate zero eligibility traces for diagonal streaming.

    Traces carried across chunk boundaries (all [B, H]):
      - Dynamics: E_nu^{c1}, E_nu^{c2}, E_th^{c1}, E_th^{c2}
      - Input w1:  E_w1^{c1}, E_w1^{c2}
      - Input w2:  E_w2^{c1}, E_w2^{c2}
    """
    E_nu_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_nu_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    E_w1_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_w1_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    E_w2_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_w2_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_w1_c1,
        E_w1_c2,
        E_w2_c1,
        E_w2_c2,
    )


def _unpack_traces_diag(trace: tuple[torch.Tensor, ...]):
    (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_w1_c1,
        E_w1_c2,
        E_w2_c1,
        E_w2_c2,
    ) = trace
    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_w1_c1,
        E_w1_c2,
        E_w2_c1,
        E_w2_c2,
    )


class _LinearRTUFunctionDiag_Streaming(Function):
    @staticmethod
    def forward(
        ctx,
        x_btd: torch.Tensor,  # (B,T,D)
        nu_log: torch.Tensor,  # (H,)
        theta_log: torch.Tensor,  # (H,)
        w1: torch.Tensor,  # (H,)
        w2: torch.Tensor,  # (H,)
        activation_name: str,
        hc1_init_bh: torch.Tensor,  # (B,H)
        hc2_init_bh: torch.Tensor,  # (B,H)
        trace_in: Optional[tuple[torch.Tensor, ...]] = None,
        resets_bt: Optional[torch.Tensor] = None,  # (B,T) bool or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        device = x_btd.device
        dtype = x_btd.dtype

        if D != H:
            raise ValueError(f"Diagonal RTU: expected D==H for identity input map, got D={D}, H={H}.")
        # Identity input map
        xhat_bth = x_btd

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

        # Input injections (diagonal per-channel weights)
        w1_1h = w1.view(1, 1, H)
        w2_1h = w2.view(1, 1, H)
        u1_bth = xhat_bth * w1_1h
        u2_bth = xhat_bth * w2_1h

        # Initialize carried traces
        if trace_in is None:
            (
                E_nu_c1,
                E_nu_c2,
                E_th_c1,
                E_th_c2,
                E_w1_c1,
                E_w1_c2,
                E_w2_c1,
                E_w2_c2,
            ) = _zeros_like_traces_diag(B, H, device=device, dtype=dtype)
        else:
            (
                E_nu_c1,
                E_nu_c2,
                E_th_c1,
                E_th_c2,
                E_w1_c1,
                E_w1_c2,
                E_w2_c1,
                E_w2_c2,
            ) = _unpack_traces_diag(trace_in)

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
                # Zero traces and states on resets (row-wise)
                E_w1_c1 = E_w1_c1 * (1.0 - m_b11.view(B, 1))
                E_w1_c2 = E_w1_c2 * (1.0 - m_b11.view(B, 1))
                E_w2_c1 = E_w2_c1 * (1.0 - m_b11.view(B, 1))
                E_w2_c2 = E_w2_c2 * (1.0 - m_b11.view(B, 1))

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

            # Update carried traces
            xhat_t = xhat_bth[:, t, :]  # (B,H)

            # w1 traces (input injected to c1)
            Ew11, Ew12 = E_w1_c1, E_w1_c2
            E_w1_c1 = g_1h.squeeze(1) * Ew11 - phi_1h.squeeze(1) * Ew12 + gamma_1h.squeeze(1) * xhat_t
            E_w1_c2 = g_1h.squeeze(1) * Ew12 + phi_1h.squeeze(1) * Ew11

            # w2 traces (input injected to c2)
            Ew21, Ew22 = E_w2_c1, E_w2_c2
            E_w2_c2 = g_1h.squeeze(1) * Ew22 + phi_1h.squeeze(1) * Ew21 + gamma_1h.squeeze(1) * xhat_t
            E_w2_c1 = g_1h.squeeze(1) * Ew21 - phi_1h.squeeze(1) * Ew22

            # Diagonal parameter traces (nu/theta)
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
            E_w1_c1,
            E_w1_c2,
            E_w2_c1,
            E_w2_c2,
        )

        # Save for backward
        if trace_in is None:
            trace_in = _zeros_like_traces_diag(B, H, device=device, dtype=dtype)
        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
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
        grad_trace_out: Optional[tuple[torch.Tensor, ...]],  # None; no grads through traces
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        (
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
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
            E_w1_c1_in,
            E_w1_c2_in,
            E_w2_c1_in,
            E_w2_c2_in,
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

        # Precompute direct drives
        # Identity map: xhat = x
        xhat_bth = x_btd

        # Reverse-time suffix scan for adjoints
        grad_x_btd = torch.zeros_like(x_btd)
        grad_w1_h = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
        grad_w2_h = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
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

            s1g = s1 * gamma_bh
            s2g = s2 * gamma_bh

            xhat_t = xhat_bth[:, t, :]
            # Local param grads for w1/w2
            grad_w1_h += torch.sum(s1g * xhat_t, dim=0)
            grad_w2_h += torch.sum(s2g * xhat_t, dim=0)

            # Input gradients
            grad_xhat_t = s1g * w1.view(1, H) + s2g * w2.view(1, H)  # (B,H)
            grad_x_btd[:, t, :] += grad_xhat_t

            # Diagonal grads accumulators using c_prev and u_t
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

            u1_t = xhat_t * w1.view(1, H)
            u2_t = xhat_t * w2.view(1, H)

            dg_sum += torch.sum(s1 * cprev1 + s2 * cprev2, dim=0)
            dphi_sum += torch.sum(-s1 * cprev2 + s2 * cprev1, dim=0)
            dgamma_sum += torch.sum(s1 * u1_t + s2 * u2_t, dim=0)

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

        # Diagonal dynamics parameter grads via reverse-mode accumulators
        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        gamma_ = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        grad_nu_log_h = -exp_nu_log * (dg_sum * g + dphi_sum * phi) + exp_nu_log * (r_val * r_val / gamma_) * dgamma_sum
        grad_theta_log_h = exp_th_log * (-dg_sum * phi + dphi_sum * g)

        # ---- Boundary corrections ----
        # Add cross-chunk boundary terms for parameters whose effects prior to
        # the chunk head influence the chunk via the carried state. For diagonal
        # input weights (w1, w2), the parameter appears at every timestep; the
        # local per-timestep gradient uses the reverse-time suffix s_t, but when
        # chunking we detach the state at the head, which removes the influence
        # of pre-chunk injections on post-chunk losses. We restore that missing
        # contribution with a single boundary addition using the carried-in
        # traces and the head-adjoint at the previous state.
        grad_w1_h = grad_w1_h + torch.sum(
            lam_prev_c1 * E_w1_c1_in + lam_prev_c2 * E_w1_c2_in,
            dim=0,
        )
        grad_w2_h = grad_w2_h + torch.sum(
            lam_prev_c1 * E_w2_c1_in + lam_prev_c2 * E_w2_c2_in,
            dim=0,
        )

        grad_nu_log_h = grad_nu_log_h + torch.sum(lam_prev_c1 * E_nu_c1_in + lam_prev_c2 * E_nu_c2_in, dim=0)
        grad_theta_log_h = grad_theta_log_h + torch.sum(lam_prev_c1 * E_th_c1_in + lam_prev_c2 * E_th_c2_in, dim=0)

        # Initial-state grads (to enable gradient flow across chunks if not detached)
        grad_hc1_init = lam_prev_c1
        grad_hc2_init = lam_prev_c2

        # No grads for activation name, P, trace_in, resets
        return (
            grad_x_btd,  # x
            grad_nu_log_h,  # nu_log
            grad_theta_log_h,  # theta_log
            grad_w1_h,  # w1
            grad_w2_h,  # w2
            None,  # activation_name
            grad_hc1_init,
            grad_hc2_init,
            None,  # trace_in
            None,  # resets
        )


def rtu_stream_diag_pytorch(
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
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Streaming functional RTU (diagonal input maps) that carries traces across chunks.

    Args:
        x_btd: [B, T, D] input (assumes D == H; identity input map)
        nu_log, theta_log: dynamics parameters [H]
        w1, w2: per-channel input weights [H]
        activation_name: e.g., "SiLU"
        hc1_init_bh, hc2_init_bh: initial states [B, H]
        trace_in: optional carried traces (E_*) as returned previously
        resets_bt: optional reset mask [B, T] or [B]

    Returns:
        y_btd_2h, (final_hc1_bh, final_hc2_bh), trace_out
    """
    y, h1, h2, trace_out = _LinearRTUFunctionDiag_Streaming.apply(
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
    return y, (h1, h2), trace_out


__all__ = ["rtu_stream_diag_pytorch"]
