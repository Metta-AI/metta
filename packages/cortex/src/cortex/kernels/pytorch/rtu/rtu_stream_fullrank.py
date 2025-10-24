"""Streaming (chunk-wise) PyTorch kernels for RTUs — FULL-RANK input maps.

This module supports processing long sequences in chunks while allowing
detached hidden states between chunks without losing cross-boundary credit
assignment. It carries forward-mode eligibility traces across subsequences
and adds a single boundary correction in the autograd backward using the
chunk-head adjoint.

Differences vs. diagonal variant:
  - Wc1, Wc2 are full-rank matrices (D×H), not diagonal vectors (H).
  - Eligibility traces E_*^{c*} for input maps are (B, D, H) instead of (B, H).

Return values: outputs, final state, and ``trace_out``. For streaming, call per
chunk, detach (hc1, hc2) and the returned ``trace_out``, and feed both back in
for the next chunk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd import Function


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


def _zeros_like_traces_full(B: int, D: int, H: int, *, device, dtype) -> tuple[torch.Tensor, ...]:
    """Allocate zero eligibility traces for full-rank streaming.

    Shapes:
      - Dynamics traces (nu/theta): all [B, H]
      - Input-map traces (Wc1, Wc2): all [B, D, H]
    """
    E_nu_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_nu_c2 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c1 = torch.zeros(B, H, device=device, dtype=dtype)
    E_th_c2 = torch.zeros(B, H, device=device, dtype=dtype)

    E_W1_c1 = torch.zeros(B, D, H, device=device, dtype=dtype)
    E_W1_c2 = torch.zeros(B, D, H, device=device, dtype=dtype)
    E_W2_c1 = torch.zeros(B, D, H, device=device, dtype=dtype)
    E_W2_c2 = torch.zeros(B, D, H, device=device, dtype=dtype)

    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_W1_c1,
        E_W1_c2,
        E_W2_c1,
        E_W2_c2,
    )


def _unpack_traces_full(trace: tuple[torch.Tensor, ...]):
    (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_W1_c1,
        E_W1_c2,
        E_W2_c1,
        E_W2_c2,
    ) = trace
    return (
        E_nu_c1,
        E_nu_c2,
        E_th_c1,
        E_th_c2,
        E_W1_c1,
        E_W1_c2,
        E_W2_c1,
        E_W2_c2,
    )


class _LinearRTUFunctionFull_Streaming(Function):
    @staticmethod
    def forward(
        ctx,
        x_btd: torch.Tensor,  # (B,T,D)
        nu_log: torch.Tensor,  # (H,)
        theta_log: torch.Tensor,  # (H,)
        Wc1: torch.Tensor,  # (D,H)
        Wc2: torch.Tensor,  # (D,H)
        activation_name: str,
        hc1_init_bh: torch.Tensor,  # (B,H)
        hc2_init_bh: torch.Tensor,  # (B,H)
        trace_in: Optional[tuple[torch.Tensor, ...]] = None,
        resets_bt: Optional[torch.Tensor] = None,  # (B,T) bool or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, T, D = x_btd.shape
        H = nu_log.shape[0]
        if Wc1.shape != (D, H) or Wc2.shape != (D, H):
            raise ValueError(f"Full-rank RTU: expected Wc* of shape (D,H); got {Wc1.shape}, {Wc2.shape}.")

        device = x_btd.device
        dtype = x_btd.dtype

        # Decode diagonal dynamics
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        theta = torch.exp(theta_log)  # (H,)
        g = r * torch.cos(theta)  # (H,)
        phi = r * torch.sin(theta)  # (H,)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))  # (H,)

        # Preallocate preactivations (carry-saving)
        pre1_bth = x_btd.new_empty(B, T, H)
        pre2_bth = x_btd.new_empty(B, T, H)

        hc1 = hc1_init_bh  # (B,H)
        hc2 = hc2_init_bh  # (B,H)

        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=device)

        # Initialize carried traces (full-rank shapes)
        if trace_in is None:
            trace = _zeros_like_traces_full(B, D, H, device=device, dtype=dtype)
        else:
            trace = _unpack_traces_full(trace_in)
        (
            E_nu_c1,
            E_nu_c2,
            E_th_c1,
            E_th_c2,
            E_W1_c1,
            E_W1_c2,
            E_W2_c1,
            E_W2_c2,
        ) = trace

        # Precompute derivatives wrt (nu_log, theta_log) for dynamics traces
        exp_nu_log = torch.exp(nu_log)  # (H,)
        exp_th_log = torch.exp(theta_log)  # (H,)
        r_val = torch.exp(-torch.exp(nu_log))  # (H,)
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        d_g_d_nu = -exp_nu_log * g  # (H,)
        d_phi_d_nu = -exp_nu_log * phi  # (H,)
        d_gamma_d_nu = exp_nu_log * r_val * r_val / torch.clamp(sqrt_1_minus_r2, min=1e-12)
        d_g_d_th = -phi * exp_th_log
        d_phi_d_th = g * exp_th_log

        g_1h = g.view(1, 1, H)  # (1,1,H) for broadcasting
        phi_1h = phi.view(1, 1, H)
        gamma_1h = gamma.view(1, 1, H)

        d_g_d_nu_bh = d_g_d_nu.view(1, H)  # (1,H) for rowwise ops
        d_phi_d_nu_bh = d_phi_d_nu.view(1, H)
        d_gamma_d_nu_bh = d_gamma_d_nu.view(1, H)
        d_g_d_th_bh = d_g_d_th.view(1, H)
        d_phi_d_th_bh = d_phi_d_th.view(1, H)

        for t in range(T):
            # Handle optional per-row resets
            if resets_bt.any():
                m_b1 = resets_bt[:, t].view(B, 1).to(dtype=dtype)
                one_minus_b1 = 1.0 - m_b1
                # zero states and dynamics traces where reset==True
                hc1 = hc1 * one_minus_b1
                hc2 = hc2 * one_minus_b1
                E_nu_c1 = E_nu_c1 * one_minus_b1
                E_nu_c2 = E_nu_c2 * one_minus_b1
                E_th_c1 = E_th_c1 * one_minus_b1
                E_th_c2 = E_th_c2 * one_minus_b1
                # zero input-map traces (full-rank) for those rows
                mask_b11 = one_minus_b1.view(B, 1, 1)
                E_W1_c1 = E_W1_c1 * mask_b11
                E_W1_c2 = E_W1_c2 * mask_b11
                E_W2_c1 = E_W2_c1 * mask_b11
                E_W2_c2 = E_W2_c2 * mask_b11

            x_t = x_btd[:, t, :]  # (B,D)

            # Input injections (full rank): u1 = x_t @ Wc1, u2 = x_t @ Wc2
            u1_t = x_t @ Wc1  # (B,H)
            u2_t = x_t @ Wc2  # (B,H)

            # Linear preactivations
            c1_t = g * hc1 - phi * hc2 + gamma * u1_t  # (B,H)
            c2_t = g * hc2 + phi * hc1 + gamma * u2_t
            pre1_bth[:, t, :] = c1_t
            pre2_bth[:, t, :] = c2_t
            hc1, hc2 = c1_t, c2_t

            # ---- Update carried input-map traces (full-rank) ----
            # E_W1^{c1} and E_W1^{c2} are (B, D, H)
            Ew11, Ew12 = E_W1_c1, E_W1_c2
            inj_1 = x_t.unsqueeze(-1) * gamma_1h.squeeze(0)  # (B,D,H)
            E_W1_c1 = Ew11 * g_1h.squeeze(0) - Ew12 * phi_1h.squeeze(0) + inj_1
            E_W1_c2 = Ew12 * g_1h.squeeze(0) + Ew11 * phi_1h.squeeze(0)

            # E_W2^{c1} and E_W2^{c2}
            Ew21, Ew22 = E_W2_c1, E_W2_c2
            inj_2 = x_t.unsqueeze(-1) * gamma_1h.squeeze(0)  # (B,D,H)
            E_W2_c2 = Ew22 * g_1h.squeeze(0) + Ew21 * phi_1h.squeeze(0) + inj_2
            E_W2_c1 = Ew21 * g_1h.squeeze(0) - Ew22 * phi_1h.squeeze(0)

            # ---- Update carried dynamics traces (nu, theta) ----
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
            E_W1_c1,
            E_W1_c2,
            E_W2_c1,
            E_W2_c2,
        )

        # Save for backward (also save carried-in traces for boundary correction)
        if trace_in is None:
            trace_in = _zeros_like_traces_full(B, D, H, device=device, dtype=dtype)
        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            Wc1,
            Wc2,
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
        grad_y_btd_2h: torch.Tensor,  # (B, T, 2H)
        grad_final_hc1: torch.Tensor,  # unused
        grad_final_hc2: torch.Tensor,  # unused
        grad_trace_out: Optional[tuple[torch.Tensor, ...]],  # None; no grads through traces
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        (
            x_btd,
            nu_log,
            theta_log,
            Wc1,
            Wc2,
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
            E_W1_c1_in,
            E_W1_c2_in,
            E_W2_c1_in,
            E_W2_c2_in,
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

        # Reverse-time suffix scan for adjoints
        grad_x_btd = torch.zeros_like(x_btd)  # (B,T,D)
        grad_Wc1 = torch.zeros_like(Wc1)  # (D,H)
        grad_Wc2 = torch.zeros_like(Wc2)  # (D,H)

        # Accumulators for dynamics grads via reverse-mode
        dg_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
        dphi_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)
        dgamma_sum = torch.zeros(H, device=x_btd.device, dtype=x_btd.dtype)

        s1_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        s2_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)

        g_bh = g.view(1, H)
        phi_bh = phi.view(1, H)
        gamma_bh = gamma.view(1, H)

        for t in range(T - 1, -1, -1):
            d1_t = eta1[:, t, :]  # (B,H)
            d2_t = eta2[:, t, :]

            if resets_bt.any():
                # Important: whether c_{t+1} depends on h_t is determined by reset at t+1
                # (forward zeroes state BEFORE computing c_{t+1} when resets[t+1] is True).
                if t + 1 < T:
                    m_next = resets_bt[:, t + 1].view(B, 1).to(dtype=d1_t.dtype)
                else:
                    m_next = torch.zeros(B, 1, dtype=d1_t.dtype, device=d1_t.device)
                one_minus = 1.0 - m_next
                s1 = d1_t + one_minus * (g_bh * s1_next + phi_bh * s2_next)
                s2 = d2_t + one_minus * (g_bh * s2_next - phi_bh * s1_next)
            else:
                s1 = d1_t + g_bh * s1_next + phi_bh * s2_next
                s2 = d2_t + g_bh * s2_next - phi_bh * s1_next

            s1g = s1 * gamma_bh  # (B,H)
            s2g = s2 * gamma_bh

            x_t = x_btd[:, t, :]  # (B,D)
            # Full-rank input-map grads (reverse-mode)
            grad_Wc1 = grad_Wc1 + x_t.transpose(0, 1) @ s1g  # (D,H)
            grad_Wc2 = grad_Wc2 + x_t.transpose(0, 1) @ s2g

            # Input gradients
            grad_x_t = s1g @ Wc1.transpose(0, 1) + s2g @ Wc2.transpose(0, 1)  # (B,D)
            grad_x_btd[:, t, :] = grad_x_btd[:, t, :] + grad_x_t

            # Dynamics accumulators using c_prev and u_t
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

            u1_t = x_t @ Wc1  # (B,H)
            u2_t = x_t @ Wc2

            dg_sum = dg_sum + torch.sum(s1 * cprev1 + s2 * cprev2, dim=0)
            dphi_sum = dphi_sum + torch.sum(-s1 * cprev2 + s2 * cprev1, dim=0)
            dgamma_sum = dgamma_sum + torch.sum(s1 * u1_t + s2 * u2_t, dim=0)

            s1_next, s2_next = s1, s2

        # Boundary adjoint (map suffix adjoint at chunk head back one step)
        lambda0_c1 = s1_next  # λ at c_{t0}
        lambda0_c2 = s2_next
        lam_prev_c1 = g.view(1, H) * lambda0_c1 + phi.view(1, H) * lambda0_c2
        lam_prev_c2 = -phi.view(1, H) * lambda0_c1 + g.view(1, H) * lambda0_c2
        if resets_bt.any():
            head_mask = 1.0 - resets_bt[:, 0].view(B, 1).to(dtype=lambda0_c1.dtype)
            lam_prev_c1 = lam_prev_c1 * head_mask
            lam_prev_c2 = lam_prev_c2 * head_mask

        # Dynamics parameter grads from accumulators
        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        gamma_ = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        grad_nu_log_h = (
            -exp_nu_log * (dg_sum * g + dphi_sum * phi)
            + exp_nu_log * (r_val * r_val / torch.clamp(gamma_, min=1e-12)) * dgamma_sum
        )
        grad_theta_log_h = exp_th_log * (-dg_sum * phi + dphi_sum * g)

        # ---- Boundary corrections (cross-chunk) ----
        lam_prev_c1_b1h = lam_prev_c1.unsqueeze(1)  # (B,1,H)
        lam_prev_c2_b1h = lam_prev_c2.unsqueeze(1)  # (B,1,H)

        grad_Wc1 = grad_Wc1 + (E_W1_c1_in * lam_prev_c1_b1h).sum(dim=0) + (E_W1_c2_in * lam_prev_c2_b1h).sum(dim=0)
        grad_Wc2 = grad_Wc2 + (E_W2_c1_in * lam_prev_c1_b1h).sum(dim=0) + (E_W2_c2_in * lam_prev_c2_b1h).sum(dim=0)

        grad_nu_log_h = grad_nu_log_h + torch.sum(lam_prev_c1 * E_nu_c1_in + lam_prev_c2 * E_nu_c2_in, dim=0)
        grad_theta_log_h = grad_theta_log_h + torch.sum(lam_prev_c1 * E_th_c1_in + lam_prev_c2 * E_th_c2_in, dim=0)

        # Initial-state grads (to enable across-chunk gradient if not detached)
        grad_hc1_init = lam_prev_c1
        grad_hc2_init = lam_prev_c2

        # No grads for activation name, trace_in, resets
        return (
            grad_x_btd,  # x
            grad_nu_log_h,  # nu_log
            grad_theta_log_h,  # theta_log
            grad_Wc1,  # Wc1
            grad_Wc2,  # Wc2
            None,  # activation_name
            grad_hc1_init,
            grad_hc2_init,
            None,  # trace_in
            None,  # resets
        )


def rtu_stream_full_pytorch(
    *,
    x_btd: torch.Tensor,  # (B,T,D)
    nu_log: torch.Tensor,  # (H,)
    theta_log: torch.Tensor,  # (H,)
    Wc1: torch.Tensor,  # (D,H)
    Wc2: torch.Tensor,  # (D,H)
    activation_name: str,
    hc1_init_bh: torch.Tensor,  # (B,H)
    hc2_init_bh: torch.Tensor,  # (B,H)
    trace_in: Optional[tuple[torch.Tensor, ...]] = None,  # carried traces (E_*) from previous chunk
    resets_bt: Optional[torch.Tensor] = None,  # (B,T) or (B)
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Streaming functional RTU (FULL-RANK input maps) that carries traces across chunks."""
    y, h1, h2, trace_out = _LinearRTUFunctionFull_Streaming.apply(
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
    return y, (h1, h2), trace_out


__all__ = ["rtu_stream_full_pytorch"]
