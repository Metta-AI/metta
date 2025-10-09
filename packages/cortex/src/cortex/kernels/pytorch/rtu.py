"""PyTorch kernel for low-rank Recurrent Trace Units (RTUs).

Custom autograd uses RTRL for diagonal dynamics and a reverse-time scan for
low-rank input maps.

Public API (functional): ``rtu_sequence_pytorch``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd import Function


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


# ---- Low-rank RTU: custom Function (RTRL for diag params; reverse-scan for low-rank inputs) ----
class _LinearRTUFunctionLR(Function):
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
        resets_bt: Optional[torch.Tensor] = None,  # (B,T) bool or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Low-rank Linear RTU forward.

        Input maps: Wc1 = U1 @ V1 (D x H), Wc2 = U2 @ V2 (D x H).
        Saves tensors needed to:
          - do RTRL for (nu_log, theta_log),
          - do a reverse-time scan for (U1, U2, V),
          - compute exact dL/dx via the reverse scan.
        """
        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        # Decode diagonal dynamics
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        theta = torch.exp(theta_log)  # (H,)
        g = r * torch.cos(theta)  # (H,)
        phi = r * torch.sin(theta)  # (H,)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))  # (H,)

        pre1_bth = x_btd.new_empty(B, T, H)
        pre2_bth = x_btd.new_empty(B, T, H)

        hc1 = hc1_init_bh
        hc2 = hc2_init_bh
        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=x_btd.device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=x_btd.device)

        a1_bth = x_btd @ U1  # (B,T,D) @ (D,R) = (B,T,R)
        a2_bth = x_btd @ U2  # (B,T,R)
        u1_bth = a1_bth @ V1  # (B,T,R) @ (R,H) = (B,T,H)
        u2_bth = a2_bth @ V2  # (B,T,H)

        for t in range(T):
            if resets_bt.any():
                m = resets_bt[:, t].view(B, 1).to(dtype=hc1.dtype)
                one_minus = 1.0 - m
                hc1 = hc1 * one_minus
                hc2 = hc2 * one_minus
            u1_bt_h = u1_bth[:, t, :]  # (B,H)
            u2_bt_h = u2_bth[:, t, :]  # (B,H)

            c1_t = g * hc1 - phi * hc2 + gamma * u1_bt_h
            c2_t = g * hc2 + phi * hc1 + gamma * u2_bt_h
            pre1_bth[:, t, :] = c1_t
            pre2_bth[:, t, :] = c2_t
            hc1, hc2 = c1_t, c2_t  # linear RTU stores c's

        y1_bth, _ = _act_and_deriv(pre1_bth, activation_name)
        y2_bth, _ = _act_and_deriv(pre2_bth, activation_name)
        y_btd_2h = torch.cat([y1_bth, y2_bth], dim=-1)  # (B,T,2H)

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
            resets_bt,
        )
        ctx.activation_name = activation_name

        # Final hidden states
        final_hc1_bh = pre1_bth[:, -1, :] if T > 0 else hc1_init_bh
        final_hc2_bh = pre2_bth[:, -1, :] if T > 0 else hc2_init_bh
        return y_btd_2h, final_hc1_bh, final_hc2_bh

    @staticmethod
    def backward(
        ctx,
        grad_y_btd_2h: torch.Tensor,  # (B,T,2H)
        grad_final_hc1: torch.Tensor,  # (B,H) (unused)
        grad_final_hc2: torch.Tensor,  # (B,H) (unused)
    ) -> tuple[Optional[torch.Tensor], ...]:
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
            resets_bt,
        ) = ctx.saved_tensors
        activation_name = ctx.activation_name

        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        _, dact1_bth = _act_and_deriv(pre1_bth, activation_name)  # (B,T,H)
        _, dact2_bth = _act_and_deriv(pre2_bth, activation_name)  # (B,T,H)

        gy1_bth = grad_y_btd_2h[:, :, :H]
        gy2_bth = grad_y_btd_2h[:, :, H:]
        dL_dc1_bth = gy1_bth * dact1_bth
        dL_dc2_bth = gy2_bth * dact2_bth

        # -----------------------------
        # Part A) RTRL for (nu_log, theta_log)
        # -----------------------------
        # Recompute a1/a2 and u1/u2 on the fly for the RTRL direct terms
        a1_btr = torch.einsum("btd,dr->btr", x_btd, U1)  # (B,T,R)
        a2_btr = torch.einsum("btd,dr->btr", x_btd, U2)  # (B,T,R)
        u1_bth = torch.einsum("btr,rh->bth", a1_btr, V1)  # (B,T,H)
        u2_bth = torch.einsum("btr,rh->bth", a2_btr, V2)  # (B,T,H)

        # Chain-rule partials wrt nu_log, theta_log
        exp_nu_log = torch.exp(nu_log)  # (H,)
        exp_th_log = torch.exp(theta_log)  # (H,)
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        d_g_d_nu = -exp_nu_log * g  # ∂g/∂nu_log
        d_phi_d_nu = -exp_nu_log * phi  # ∂phi/∂nu_log
        d_gamma_d_nu = exp_nu_log * r * r / sqrt_1_minus_r2  # ∂γ/∂nu_log

        d_g_d_th = -phi * exp_th_log  # ∂g/∂theta_log
        d_phi_d_th = g * exp_th_log  # ∂phi/∂theta_log

        E_nu_c1_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        E_nu_c2_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        E_th_c1_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        E_th_c2_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)

        grad_nu_log_h = torch.zeros_like(nu_log)
        grad_theta_log_h = torch.zeros_like(theta_log)

        hc1_prev = hc1_init_bh
        hc2_prev = hc2_init_bh

        g_bh = g.view(1, H)
        phi_bh = phi.view(1, H)
        d_g_d_nu_bh = d_g_d_nu.view(1, H)
        d_phi_d_nu_bh = d_phi_d_nu.view(1, H)
        d_gamma_d_nu_bh = d_gamma_d_nu.view(1, H)
        d_g_d_th_bh = d_g_d_th.view(1, H)
        d_phi_d_th_bh = d_phi_d_th.view(1, H)

        for t in range(T):
            c1_t = pre1_bth[:, t, :]  # (B,H)
            c2_t = pre2_bth[:, t, :]  # (B,H)
            u1_t = u1_bth[:, t, :]  # (B,H)
            u2_t = u2_bth[:, t, :]  # (B,H)

            if resets_bt.any():
                # Zero out effect of previous state and traces when reset at t
                m = resets_bt[:, t].view(B, 1).to(dtype=hc1_prev.dtype)
                one_minus = 1.0 - m
                hc1_prev = hc1_prev * one_minus
                hc2_prev = hc2_prev * one_minus
                Enu_c1_old = E_nu_c1_bh * one_minus
                Enu_c2_old = E_nu_c2_bh * one_minus
                Eth_c1_old = E_th_c1_bh * one_minus
                Eth_c2_old = E_th_c2_bh * one_minus
            else:
                Enu_c1_old = E_nu_c1_bh
                Enu_c2_old = E_nu_c2_bh
                Eth_c1_old = E_th_c1_bh
                Eth_c2_old = E_th_c2_bh

            E_nu_c1_bh = (
                d_g_d_nu_bh * hc1_prev
                + g_bh * Enu_c1_old
                - d_phi_d_nu_bh * hc2_prev
                - phi_bh * Enu_c2_old
                + d_gamma_d_nu_bh * u1_t
            )
            E_nu_c2_bh = (
                d_g_d_nu_bh * hc2_prev
                + g_bh * Enu_c2_old
                + d_phi_d_nu_bh * hc1_prev
                + phi_bh * Enu_c1_old
                + d_gamma_d_nu_bh * u2_t
            )

            E_th_c1_bh = d_g_d_th_bh * hc1_prev + g_bh * Eth_c1_old - d_phi_d_th_bh * hc2_prev - phi_bh * Eth_c2_old
            E_th_c2_bh = d_g_d_th_bh * hc2_prev + g_bh * Eth_c2_old + d_phi_d_th_bh * hc1_prev + phi_bh * Eth_c1_old

            # Accumulate grads to nu/theta using updated traces
            grad_nu_log_h += torch.sum(
                dL_dc1_bth[:, t, :] * E_nu_c1_bh + dL_dc2_bth[:, t, :] * E_nu_c2_bh,
                dim=0,
            )
            grad_theta_log_h += torch.sum(
                dL_dc1_bth[:, t, :] * E_th_c1_bh + dL_dc2_bth[:, t, :] * E_th_c2_bh,
                dim=0,
            )

            hc1_prev = c1_t
            hc2_prev = c2_t

        grad_x_btd = torch.zeros_like(x_btd)  # (B,T,D)
        GW1_DH = torch.zeros(D, H, device=x_btd.device, dtype=x_btd.dtype)
        GW2_DH = torch.zeros(D, H, device=x_btd.device, dtype=x_btd.dtype)

        s1_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        s2_next = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)

        g_bh = g.view(1, H)
        phi_bh = phi.view(1, H)
        gamma_bh = gamma.view(1, H)

        for t in range(T - 1, -1, -1):
            d1 = dL_dc1_bth[:, t, :]
            d2 = dL_dc2_bth[:, t, :]
            if resets_bt.any():
                m = resets_bt[:, t].view(B, 1).to(dtype=d1.dtype)
                one_minus = 1.0 - m
                s1 = d1 + one_minus * (g_bh * s1_next + phi_bh * s2_next)
                s2 = d2 + one_minus * (g_bh * s2_next - phi_bh * s1_next)
            else:
                s1 = d1 + g_bh * s1_next + phi_bh * s2_next
                s2 = d2 + g_bh * s2_next - phi_bh * s1_next

            s1g = s1 * gamma_bh  # (B,H)
            s2g = s2 * gamma_bh

            X_t = x_btd[:, t, :]  # (B,D)
            GW1_DH += X_t.t() @ s1g  # (D,H)
            GW2_DH += X_t.t() @ s2g  # (D,H)

            tmp1_BR = s1g @ V1.t()  # (B,R)
            tmp2_BR = s2g @ V2.t()  # (B,R)
            grad_x_btd[:, t, :] += tmp1_BR @ U1.t() + tmp2_BR @ U2.t()

            s1_next, s2_next = s1, s2

        grad_U1_DR = GW1_DH @ V1.t()  # (D,H) @ (H,R) -> (D,R)
        grad_U2_DR = GW2_DH @ V2.t()  # (D,R)
        grad_V1_RH = U1.t() @ GW1_DH  # (R,D)@(D,H) -> (R,H)
        grad_V2_RH = U2.t() @ GW2_DH

        # Return non-zero grads for initial state so gradients can flow
        # across subsequences (chunked BPTT / streaming).
        grad_hc1_init = s1_next
        grad_hc2_init = s2_next

        return (
            grad_x_btd,  # x
            grad_nu_log_h,  # nu_log
            grad_theta_log_h,  # theta_log
            grad_U1_DR,  # U1
            grad_U2_DR,  # U2
            grad_V1_RH,  # V1
            grad_V2_RH,  # V2
            None,  # activation_name (str)
            grad_hc1_init,  # hc1_init
            grad_hc2_init,  # hc2_init
            None,  # resets
        )


def rtu_sequence_pytorch(
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
    resets_bt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Functional RTU interface (PyTorch autograd).

    Args:
        x_btd: Input [B, T, D]
        nu_log, theta_log: dynamics parameters [H]
        U1, U2: [D, R]
        V1, V2: [R, H]
        activation_name: e.g., "SiLU"
        hc1_init_bh, hc2_init_bh: initial states [B, H]
        resets_bt: optional reset mask [B, T] or [B]

    Returns:
        y_btd_2h, (final_hc1_bh, final_hc2_bh)
    """
    y_btd_2h, final_hc1_bh, final_hc2_bh = _LinearRTUFunctionLR.apply(
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
        resets_bt,
    )
    return y_btd_2h, (final_hc1_bh, final_hc2_bh)


__all__ = ["rtu_sequence_pytorch"]
