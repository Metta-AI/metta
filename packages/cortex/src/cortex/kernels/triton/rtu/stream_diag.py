from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd import Function

try:
    import triton as _triton  # type: ignore  # noqa: F401
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False

from .utils import hillis_steele_segmented_inplace, scan_step_block_segmented


def _act_and_deriv(z: torch.Tensor, activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    name = activation.lower()
    if name in ["silu", "swish"]:
        y = torch.nn.functional.silu(z)
        s = torch.sigmoid(z)
        dy = s * (1.0 + z * (1.0 - s))
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


class _RTUStreamDiagFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x_btd: torch.Tensor,  # (B,T,D) with D==H
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
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")

        B, T, D = x_btd.shape
        H = nu_log.shape[0]
        if D != H:
            raise ValueError(f"Diagonal RTU (Triton): expected D==H, got D={D}, H={H}.")

        device = x_btd.device
        dtype = x_btd.dtype

        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=device)

        # Dynamics decoding
        r = torch.exp(-torch.exp(nu_log))  # (H,)
        theta = torch.exp(theta_log)  # (H,)
        g = r * torch.cos(theta)
        phi = r * torch.sin(theta)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        # Identity input map with diagonal per-channel weights
        u1_bth = x_btd * w1.view(1, 1, H)
        u2_bth = x_btd * w2.view(1, 1, H)
        b_bth2 = torch.stack([u1_bth, u2_bth], dim=-1) * gamma.view(1, 1, H, 1)

        # Prepare segmented-scan buffers for c_t (pre-activations)
        g_bht = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        p_bht = phi.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        Bx_bht = b_bth2[..., 0].permute(0, 2, 1).contiguous()
        By_bht = b_bth2[..., 1].permute(0, 2, 1).contiguous()

        # Seed head with c0 contribution when not reset
        if T > 0:
            gg = g.view(1, H).expand(B, H)
            pp = phi.view(1, H).expand(B, H)
            rotx0 = gg * hc1_init_bh - pp * hc2_init_bh
            roty0 = pp * hc1_init_bh + gg * hc2_init_bh
            use_c0 = (~resets_bt[:, 0]).to(Bx_bht.dtype).view(B, 1)
            Bx_bht[:, :, 0] = Bx_bht[:, :, 0] + use_c0 * rotx0
            By_bht[:, :, 0] = By_bht[:, :, 0] + use_c0 * roty0

        flags_bht = torch.zeros(B, H, T, dtype=torch.int32, device=device)
        if T > 0:
            flags_bht[:, :, 0] = 1
        if resets_bt.any():
            flags_bht |= resets_bt.view(B, 1, T).expand(B, H, T).to(torch.int32)

        # Parallel segmented scan for c_t
        hillis_steele_segmented_inplace(g_bht, p_bht, Bx_bht, By_bht, flags_bht)
        c1_bth = Bx_bht.permute(0, 2, 1).contiguous()
        c2_bth = By_bht.permute(0, 2, 1).contiguous()

        # Activation
        y1_bth, _ = _act_and_deriv(c1_bth, activation_name)
        y2_bth, _ = _act_and_deriv(c2_bth, activation_name)
        y_btd_2h = torch.cat([y1_bth, y2_bth], dim=-1)

        # Compute outgoing carried traces via block segmented scan on augmented state
        Bhc_x = (gamma.view(1, H, 1) * u1_bth.permute(0, 2, 1)).contiguous()
        Bhc_y = (gamma.view(1, H, 1) * u2_bth.permute(0, 2, 1)).contiguous()

        if T > 0:
            gg = g.view(1, H).expand(B, H)
            pp = phi.view(1, H).expand(B, H)
            rotx0 = gg * hc1_init_bh - pp * hc2_init_bh
            roty0 = pp * hc1_init_bh + gg * hc2_init_bh
            use_c0 = (~resets_bt[:, 0]).to(Bhc_x.dtype).view(B, 1)
            Bhc_x[:, :, 0] = Bhc_x[:, :, 0] + use_c0 * rotx0
            Bhc_y[:, :, 0] = Bhc_y[:, :, 0] + use_c0 * roty0

        def run_block_scan(
            jg_param: torch.Tensor, jp_param: torch.Tensor, Bex_inj: torch.Tensor, Bey_inj: torch.Tensor
        ):
            g_bht_local = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
            p_bht_local = phi.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
            jg_bht = (
                jg_param.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
            )
            jp_bht = (
                jp_param.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
            )
            Be_x = Bex_inj.permute(0, 2, 1).contiguous()
            Be_y = Bey_inj.permute(0, 2, 1).contiguous()

            if T > 0:
                jx0 = jg_param.view(1, H).expand(B, H) * hc1_init_bh - jp_param.view(1, H).expand(B, H) * hc2_init_bh
                jy0 = jp_param.view(1, H).expand(B, H) * hc1_init_bh + jg_param.view(1, H).expand(B, H) * hc2_init_bh
                use_c0 = (~resets_bt[:, 0]).to(Be_x.dtype).view(B, 1)
                Be_x[:, :, 0] = Be_x[:, :, 0] + use_c0 * jx0
                Be_y[:, :, 0] = Be_y[:, :, 0] + use_c0 * jy0

            flags = torch.zeros(B, H, T, dtype=torch.int32, device=device)
            if T > 0:
                flags[:, :, 0] = 1
            if resets_bt.any():
                flags |= resets_bt.view(B, 1, T).expand(B, H, T).to(torch.int32)

            scan_step_block_segmented(
                g_bht_local,
                p_bht_local,
                jg_bht,
                jp_bht,
                Bhc_x.clone(),
                Bhc_y.clone(),
                Be_x,
                Be_y,
                flags,
            )
            if T > 0:
                Ex_final = Be_x[:, :, -1].contiguous()
                Ey_final = Be_y[:, :, -1].contiguous()
            else:
                Ex_final = torch.zeros(B, H, device=device, dtype=dtype)
                Ey_final = torch.zeros(B, H, device=device, dtype=dtype)
            return Ex_final, Ey_final

        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        d_g_d_nu = -exp_nu_log * g
        d_phi_d_nu = -exp_nu_log * phi
        d_gamma_d_nu = exp_nu_log * r_val * r_val / sqrt_1_minus_r2
        d_g_d_th = -phi * exp_th_log
        d_phi_d_th = g * exp_th_log

        xhat_bth = x_btd
        Bex_w1 = gamma.view(1, 1, H) * xhat_bth
        Bey_w1 = torch.zeros_like(Bex_w1)
        Bex_w2 = torch.zeros_like(Bex_w1)
        Bey_w2 = gamma.view(1, 1, H) * xhat_bth
        Bex_nu = d_gamma_d_nu.view(1, 1, H) * u1_bth
        Bey_nu = d_gamma_d_nu.view(1, 1, H) * u2_bth
        Bex_th = torch.zeros_like(Bex_w1)
        Bey_th = torch.zeros_like(Bex_w1)

        E_w1_c1, E_w1_c2 = run_block_scan(torch.zeros_like(g), torch.zeros_like(phi), Bex_w1, Bey_w1)
        E_w2_c1, E_w2_c2 = run_block_scan(torch.zeros_like(g), torch.zeros_like(phi), Bex_w2, Bey_w2)
        E_nu_c1, E_nu_c2 = run_block_scan(d_g_d_nu, d_phi_d_nu, Bex_nu, Bey_nu)
        E_th_c1, E_th_c2 = run_block_scan(d_g_d_th, d_phi_d_th, Bex_th, Bey_th)

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

        if trace_in is None:
            trace_in = _zeros_like_traces_diag(B, H, device=device, dtype=dtype)
        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
            c1_bth,
            c2_bth,
            g,
            phi,
            gamma,
            hc1_init_bh,
            hc2_init_bh,
            *trace_in,
            resets_bt,
        )
        ctx.activation_name = activation_name

        final_hc1_bh = c1_bth[:, -1, :] if T > 0 else hc1_init_bh
        final_hc2_bh = c2_bth[:, -1, :] if T > 0 else hc2_init_bh
        return y_btd_2h, final_hc1_bh, final_hc2_bh, trace_out

    @staticmethod
    def backward(
        ctx,
        grad_y_btd_2h: torch.Tensor,
        grad_final_hc1: torch.Tensor,
        grad_final_hc2: torch.Tensor,
        grad_trace_out: Optional[tuple[torch.Tensor, ...]],
    ) -> tuple[Optional[torch.Tensor], ...]:
        (
            x_btd,
            nu_log,
            theta_log,
            w1,
            w2,
            c1_bth,
            c2_bth,
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
        ) = ctx.saved_tensors
        activation_name = ctx.activation_name

        B, T, D = x_btd.shape
        H = nu_log.shape[0]

        _, d1 = _act_and_deriv(c1_bth, activation_name)
        _, d2 = _act_and_deriv(c2_bth, activation_name)
        gy1 = grad_y_btd_2h[:, :, :H]
        gy2 = grad_y_btd_2h[:, :, H:]
        eta1 = d1 * gy1
        eta2 = d2 * gy2

        eta1_rev = torch.flip(eta1, dims=[1])
        eta2_rev = torch.flip(eta2, dims=[1])
        vBx = eta1_rev.permute(0, 2, 1).contiguous()
        vBy = eta2_rev.permute(0, 2, 1).contiguous()
        g_bar = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        p_bar = (-phi).view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()

        flags_rev = torch.zeros(B, H, T, dtype=torch.int32, device=x_btd.device)
        if T > 0:
            flags_rev[:, :, 0] = 1
        if resets_bt.any():
            resets_rev = torch.flip(resets_bt, dims=[1])
            flags_rev |= resets_rev.view(B, 1, -1).expand_as(flags_rev).to(torch.int32)

        hillis_steele_segmented_inplace(g_bar, p_bar, vBx, vBy, flags_rev)
        lam1_bth = torch.flip(vBx.permute(0, 2, 1), dims=[1])
        lam2_bth = torch.flip(vBy.permute(0, 2, 1), dims=[1])

        s1g = lam1_bth * gamma.view(1, 1, H)
        s2g = lam2_bth * gamma.view(1, 1, H)
        grad_x_btd = s1g * w1.view(1, 1, H) + s2g * w2.view(1, 1, H)
        grad_w1_h = torch.sum(s1g * x_btd, dim=(0, 1))
        grad_w2_h = torch.sum(s2g * x_btd, dim=(0, 1))

        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r_val = torch.exp(-torch.exp(nu_log))
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))

        # Vectorized segmented reduction across time for dynamics parameter sums
        # Build c_{t-1} by shifting and seeding with initial state
        cprev1_bth = torch.cat([hc1_init_bh.unsqueeze(1), c1_bth[:, :-1, :]], dim=1)
        cprev2_bth = torch.cat([hc2_init_bh.unsqueeze(1), c2_bth[:, :-1, :]], dim=1)
        if resets_bt.any():
            gate_bth = (1.0 - resets_bt.to(dtype=c1_bth.dtype)).unsqueeze(-1)
            cprev1_bth = cprev1_bth * gate_bth
            cprev2_bth = cprev2_bth * gate_bth

        dg_sum = (lam1_bth * cprev1_bth + lam2_bth * cprev2_bth).sum(dim=(0, 1))
        dphi_sum = (-lam1_bth * cprev2_bth + lam2_bth * cprev1_bth).sum(dim=(0, 1))
        u1_bth = x_btd * w1.view(1, 1, H)
        u2_bth = x_btd * w2.view(1, 1, H)
        dgamma_sum = (lam1_bth * u1_bth + lam2_bth * u2_bth).sum(dim=(0, 1))

        g = r_val * torch.cos(torch.exp(theta_log))
        phi = r_val * torch.sin(torch.exp(theta_log))
        gamma = torch.sqrt(torch.clamp(1.0 - r_val * r_val, min=0.0))
        grad_nu_log_h = -exp_nu_log * (dg_sum * g + dphi_sum * phi) + exp_nu_log * (
            r_val * r_val / sqrt_1_minus_r2
        ) * dgamma_sum
        grad_theta_log_h = exp_th_log * (-dg_sum * phi + dphi_sum * g)

        lambda0_c1 = lam1_bth[:, 0, :] if T > 0 else torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        lambda0_c2 = lam2_bth[:, 0, :] if T > 0 else torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
        lam_prev_c1 = g.view(1, H) * lambda0_c1 + phi.view(1, H) * lambda0_c2
        lam_prev_c2 = -phi.view(1, H) * lambda0_c1 + g.view(1, H) * lambda0_c2
        if resets_bt.any() and T > 0:
            head_mask = 1.0 - resets_bt[:, 0].view(B, 1).to(dtype=lam_prev_c1.dtype)
            lam_prev_c1 = lam_prev_c1 * head_mask
            lam_prev_c2 = lam_prev_c2 * head_mask

        grad_w1_h = grad_w1_h + torch.sum(lam_prev_c1 * E_w1_c1_in + lam_prev_c2 * E_w1_c2_in, dim=0)
        grad_w2_h = grad_w2_h + torch.sum(lam_prev_c1 * E_w2_c1_in + lam_prev_c2 * E_w2_c2_in, dim=0)
        grad_nu_log_h = grad_nu_log_h + torch.sum(lam_prev_c1 * E_nu_c1_in + lam_prev_c2 * E_nu_c2_in, dim=0)
        grad_theta_log_h = grad_theta_log_h + torch.sum(lam_prev_c1 * E_th_c1_in + lam_prev_c2 * E_th_c2_in, dim=0)

        dhc1_init = lam_prev_c1
        dhc2_init = lam_prev_c2

        return (
            grad_x_btd,
            grad_nu_log_h,
            grad_theta_log_h,
            grad_w1_h,
            grad_w2_h,
            None,
            dhc1_init,
            dhc2_init,
            None,
            None,
        )


def rtu_stream_diag_triton(
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
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")
    y, h1, h2, trace_out = _RTUStreamDiagFunction.apply(
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

__all__ = ["rtu_stream_diag_triton"]
