"""Triton-based, time-parallel RTU (low-rank input maps) with segmented resets.

Moved from `kernels.triton.mlstm.rtu_triton` to `kernels.triton.rtu`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False


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


@triton.jit
def _scan_step_kernel(
    GPhi_g_ptr,
    GPhi_phi_ptr,
    Bx_ptr,
    By_ptr,
    T: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_t: tl.constexpr,
    offset: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_lane = tl.program_id(0)
    pid_blk = tl.program_id(1)

    H = stride_b // stride_h
    b = pid_lane // H
    h = pid_lane % H

    t_idxs = offset + pid_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_idxs < T

    base = b * stride_b + h * stride_h

    r_idx = base + t_idxs * stride_t
    g_r = tl.load(GPhi_g_ptr + r_idx, mask=mask, other=0.0)
    p_r = tl.load(GPhi_phi_ptr + r_idx, mask=mask, other=0.0)
    bx_r = tl.load(Bx_ptr + r_idx, mask=mask, other=0.0)
    by_r = tl.load(By_ptr + r_idx, mask=mask, other=0.0)

    l_t = t_idxs - offset
    l_idx = base + l_t * stride_t
    g_l = tl.load(GPhi_g_ptr + l_idx, mask=mask, other=0.0)
    p_l = tl.load(GPhi_phi_ptr + l_idx, mask=mask, other=0.0)
    bx_l = tl.load(Bx_ptr + l_idx, mask=mask, other=0.0)
    by_l = tl.load(By_ptr + l_idx, mask=mask, other=0.0)

    g_new = g_r * g_l - p_r * p_l
    p_new = p_r * g_l + g_r * p_l
    rot_lx = g_r * bx_l - p_r * by_l
    rot_ly = p_r * bx_l + g_r * by_l
    bx_new = rot_lx + bx_r
    by_new = rot_ly + by_r

    tl.store(GPhi_g_ptr + r_idx, g_new, mask=mask)
    tl.store(GPhi_phi_ptr + r_idx, p_new, mask=mask)
    tl.store(Bx_ptr + r_idx, bx_new, mask=mask)
    tl.store(By_ptr + r_idx, by_new, mask=mask)


def _hillis_steele_scan_inplace(
    gphi_g: torch.Tensor, gphi_phi: torch.Tensor, Bx: torch.Tensor, By: torch.Tensor
) -> None:
    assert gphi_g.is_contiguous() and gphi_phi.is_contiguous()
    assert Bx.is_contiguous() and By.is_contiguous()
    assert gphi_g.shape == gphi_phi.shape == Bx.shape == By.shape
    B, H, T = gphi_g.shape
    if T <= 1:
        return

    stride_t = 1
    stride_h = T
    stride_b = H * T

    BLOCK_T = 128
    offset = 1
    while offset < T:
        max_tiles = max(T - offset, 0)
        if max_tiles > 0:
            grid = (B * H, (max_tiles + BLOCK_T - 1) // BLOCK_T)
            _scan_step_kernel[grid](
                gphi_g,
                gphi_phi,
                Bx,
                By,
                T,
                stride_b,
                stride_h,
                stride_t,
                offset,
                BLOCK_T,
                num_warps=4,
            )
        offset <<= 1


@triton.jit
def _scan_step_segmented_kernel(
    GPhi_g_ptr,
    GPhi_phi_ptr,
    Bx_ptr,
    By_ptr,
    Flags_ptr,
    T: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_t: tl.constexpr,
    offset: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_lane = tl.program_id(0)
    pid_blk = tl.program_id(1)

    H = stride_b // stride_h
    b = pid_lane // H
    h = pid_lane % H

    t_idxs = offset + pid_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_idxs < T

    base = b * stride_b + h * stride_h

    r_idx = base + t_idxs * stride_t
    g_r = tl.load(GPhi_g_ptr + r_idx, mask=mask, other=0.0)
    p_r = tl.load(GPhi_phi_ptr + r_idx, mask=mask, other=0.0)
    bx_r = tl.load(Bx_ptr + r_idx, mask=mask, other=0.0)
    by_r = tl.load(By_ptr + r_idx, mask=mask, other=0.0)
    fr = tl.load(Flags_ptr + r_idx, mask=mask, other=1)

    l_t = t_idxs - offset
    l_idx = base + l_t * stride_t
    g_l = tl.load(GPhi_g_ptr + l_idx, mask=mask, other=0.0)
    p_l = tl.load(GPhi_phi_ptr + l_idx, mask=mask, other=0.0)
    bx_l = tl.load(Bx_ptr + l_idx, mask=mask, other=0.0)
    by_l = tl.load(By_ptr + l_idx, mask=mask, other=0.0)
    fl = tl.load(Flags_ptr + l_idx, mask=mask, other=1)

    g_new = g_r * g_l - p_r * p_l
    p_new = p_r * g_l + g_r * p_l
    rot_lx = g_r * bx_l - p_r * by_l
    rot_ly = p_r * bx_l + g_r * by_l
    bx_new = rot_lx + bx_r
    by_new = rot_ly + by_r

    cond = fr == 0
    g_out = tl.where(cond, g_new, g_r)
    p_out = tl.where(cond, p_new, p_r)
    bx_out = tl.where(cond, bx_new, bx_r)
    by_out = tl.where(cond, by_new, by_r)
    f_out = tl.minimum(fr + fl, 1)

    tl.store(GPhi_g_ptr + r_idx, g_out, mask=mask)
    tl.store(GPhi_phi_ptr + r_idx, p_out, mask=mask)
    tl.store(Bx_ptr + r_idx, bx_out, mask=mask)
    tl.store(By_ptr + r_idx, by_out, mask=mask)
    tl.store(Flags_ptr + r_idx, f_out, mask=mask)


def _hillis_steele_segmented_inplace(
    gphi_g: torch.Tensor,
    gphi_phi: torch.Tensor,
    Bx: torch.Tensor,
    By: torch.Tensor,
    flags: torch.Tensor,
) -> None:
    assert gphi_g.is_contiguous() and gphi_phi.is_contiguous()
    assert Bx.is_contiguous() and By.is_contiguous() and flags.is_contiguous()
    assert gphi_g.shape == gphi_phi.shape == Bx.shape == By.shape == flags.shape
    B, H, T = gphi_g.shape
    if T <= 1:
        return

    stride_t = 1
    stride_h = T
    stride_b = H * T

    BLOCK_T = 128
    offset = 1
    while offset < T:
        max_tiles = max(T - offset, 0)
        if max_tiles > 0:
            grid = (B * H, (max_tiles + BLOCK_T - 1) // BLOCK_T)
            _scan_step_segmented_kernel[grid](
                gphi_g,
                gphi_phi,
                Bx,
                By,
                flags,
                T,
                stride_b,
                stride_h,
                stride_t,
                offset,
                BLOCK_T,
                num_warps=4,
            )
        offset <<= 1


class _LinearRTUFunctionLR_Triton(Function):
    @staticmethod
    def forward(
        ctx,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")

        B, T, D = x_btd.shape
        if resets_bt is None:
            resets_bt = torch.zeros(B, T, dtype=torch.bool, device=x_btd.device)
        else:
            if resets_bt.dim() == 1:
                resets_bt = resets_bt.view(B, 1).expand(B, T)
            resets_bt = resets_bt.to(dtype=torch.bool, device=x_btd.device)

        H = nu_log.shape[0]

        r = torch.exp(-torch.exp(nu_log))
        theta = torch.exp(theta_log)
        g = r * torch.cos(theta)
        phi = r * torch.sin(theta)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        a1_btr = torch.einsum("btd,dr->btr", x_btd, U1)
        a2_btr = torch.einsum("btd,dr->btr", x_btd, U2)
        u1_bth = torch.einsum("btr,rh->bth", a1_btr, V1)
        u2_bth = torch.einsum("btr,rh->bth", a2_btr, V2)
        b_bth2 = torch.stack([u1_bth, u2_bth], dim=-1) * gamma.view(1, 1, H, 1)

        g_bht = g.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        p_bht = phi.view(1, H).expand(B, H).unsqueeze(-1).expand(B, H, T).contiguous()
        Bx_bht = b_bth2[..., 0].permute(0, 2, 1).contiguous()
        By_bht = b_bth2[..., 1].permute(0, 2, 1).contiguous()

        if T > 0:
            c0x = hc1_init_bh
            c0y = hc2_init_bh
            gg = g.view(1, H).expand(B, H)
            pp = phi.view(1, H).expand(B, H)
            rotx0 = gg * c0x - pp * c0y
            roty0 = pp * c0x + gg * c0y
            use_c0 = (~resets_bt[:, 0]).to(Bx_bht.dtype).view(B, 1)
            Bx_bht[:, :, 0] = Bx_bht[:, :, 0] + use_c0 * rotx0
            By_bht[:, :, 0] = By_bht[:, :, 0] + use_c0 * roty0

        flags_bht = torch.zeros(B, H, T, dtype=torch.int32, device=x_btd.device)
        if T > 0:
            flags_bht[:, :, 0] = 1
        if resets_bt.any():
            flags_bht |= resets_bt.view(B, 1, T).expand(B, H, T).to(torch.int32)

        _hillis_steele_segmented_inplace(g_bht, p_bht, Bx_bht, By_bht, flags_bht)

        c1_bth = Bx_bht.permute(0, 2, 1).contiguous()
        c2_bth = By_bht.permute(0, 2, 1).contiguous()

        y1_bth, _ = _act_and_deriv(c1_bth, activation_name)
        y2_bth, _ = _act_and_deriv(c2_bth, activation_name)
        y_btd_2h = torch.cat([y1_bth, y2_bth], dim=-1)

        ctx.save_for_backward(
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            c1_bth,
            c2_bth,
            g,
            phi,
            gamma,
            a1_btr,
            a2_btr,
            u1_bth,
            u2_bth,
            hc1_init_bh,
            hc2_init_bh,
            resets_bt,
        )
        ctx.activation_name = activation_name

        final_hc1_bh = c1_bth[:, -1, :] if T > 0 else hc1_init_bh
        final_hc2_bh = c2_bth[:, -1, :] if T > 0 else hc2_init_bh
        return y_btd_2h, final_hc1_bh, final_hc2_bh

    @staticmethod
    def backward(
        ctx,
        grad_y_btd_2h: torch.Tensor,
        grad_final_hc1: torch.Tensor,
        grad_final_hc2: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        (
            x_btd,
            nu_log,
            theta_log,
            U1,
            U2,
            V1,
            V2,
            c1_bth,
            c2_bth,
            g,
            phi,
            gamma,
            a1_btr,
            a2_btr,
            u1_bth,
            u2_bth,
            hc1_init_bh,
            hc2_init_bh,
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

        _hillis_steele_segmented_inplace(g_bar, p_bar, vBx, vBy, flags_rev)

        lam1_bth = torch.flip(vBx.permute(0, 2, 1), dims=[1])
        lam2_bth = torch.flip(vBy.permute(0, 2, 1), dims=[1])

        s1g = lam1_bth * gamma.view(1, 1, H)
        s2g = lam2_bth * gamma.view(1, 1, H)
        X_btD = x_btd
        GW1_DH = torch.einsum("btd,bth->dh", X_btD, s1g)
        GW2_DH = torch.einsum("btd,bth->dh", X_btD, s2g)

        tmp1_BR = torch.einsum("bth,rh->btr", s1g, V1)
        tmp2_BR = torch.einsum("bth,rh->btr", s2g, V2)
        dx_btd = torch.einsum("btr,dr->btd", tmp1_BR, U1) + torch.einsum("btr,dr->btd", tmp2_BR, U2)

        dU1_DR = torch.einsum("dh,hr->dr", GW1_DH, V1.t())
        dU2_DR = torch.einsum("dh,hr->dr", GW2_DH, V2.t())
        dV1_RH = torch.einsum("rd,dh->rh", U1.t(), GW1_DH)
        dV2_RH = torch.einsum("rd,dh->rh", U2.t(), GW2_DH)

        dL_dc1_bth = eta1
        dL_dc2_bth = eta2

        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        r = torch.exp(-torch.exp(nu_log))
        sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

        d_g_d_nu_bh = (-exp_nu_log * g).view(1, H)
        d_phi_d_nu_bh = (-exp_nu_log * phi).view(1, H)
        d_gamma_d_nu_bh = (exp_nu_log * r * r / sqrt_1_minus_r2).view(1, H)
        d_g_d_th_bh = (-phi * exp_th_log).view(1, H)
        d_phi_d_th_bh = (g * exp_th_log).view(1, H)

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

        for t in range(T):
            c1_t = c1_bth[:, t, :]
            c2_t = c2_bth[:, t, :]
            u1_t = u1_bth[:, t, :]
            u2_t = u2_bth[:, t, :]

            if resets_bt.any():
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

            grad_nu_log_h += torch.sum(dL_dc1_bth[:, t, :] * E_nu_c1_bh + dL_dc2_bth[:, t, :] * E_nu_c2_bh, dim=0)
            grad_theta_log_h += torch.sum(dL_dc1_bth[:, t, :] * E_th_c1_bh + dL_dc2_bth[:, t, :] * E_th_c2_bh, dim=0)

            hc1_prev = c1_t
            hc2_prev = c2_t

        dnu_log = grad_nu_log_h
        dth_log = grad_theta_log_h

        dhc1_init = torch.zeros_like(hc1_init_bh)
        dhc2_init = torch.zeros_like(hc2_init_bh)

        return (
            dx_btd,
            dnu_log,
            dth_log,
            dU1_DR,
            dU2_DR,
            dV1_RH,
            dV2_RH,
            None,
            dhc1_init,
            dhc2_init,
            None,
        )


class LinearRTU_Triton(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        rank: int,
        batch_first: bool = True,
        activation: Optional[nn.Module] = None,
        r_max: float = 1.0,
        r_min: float = 0.0,
        max_phase: float = 6.28,
    ) -> None:
        super().__init__()
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")
        if rank < 1 or rank > min(input_size, hidden_size):
            raise ValueError(f"rank must be in [1, min(D,H)] but got {rank}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.batch_first = batch_first
        self.activation = activation if activation is not None else nn.Identity()

        u1 = torch.rand(hidden_size)
        inner = u1 * (r_max**2 - r_min**2) + r_min**2
        nu_log_init = torch.log(-0.5 * torch.log(inner.clamp(min=1e-12)))
        u2 = torch.rand(hidden_size)
        theta_log_init = torch.log((max_phase * u2).clamp(min=1e-12))
        self.nu_log = nn.Parameter(nu_log_init)
        self.theta_log = nn.Parameter(theta_log_init)

        self.U1 = nn.Parameter(torch.empty(input_size, rank))
        self.U2 = nn.Parameter(torch.empty(input_size, rank))
        self.V1 = nn.Parameter(torch.empty(rank, hidden_size))
        self.V2 = nn.Parameter(torch.empty(rank, hidden_size))

        bound_in = 1.0 / math.sqrt(input_size)
        bound_r = 1.0 / math.sqrt(rank)
        with torch.no_grad():
            self.U1.uniform_(-bound_in, bound_in)
            self.U2.uniform_(-bound_in, bound_in)
            self.V1.uniform_(-bound_r, bound_r)
            self.V2.uniform_(-bound_r, bound_r)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        resets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, T, _ = x.shape

        if hx is None:
            hc1_init = x.new_zeros(B, self.hidden_size)
            hc2_init = x.new_zeros(B, self.hidden_size)
        else:
            hc1_init, hc2_init = hx

        act_name = self.activation.__class__.__name__
        y_btd_2h, hc1, hc2 = _LinearRTUFunctionLR_Triton.apply(
            x,
            self.nu_log,
            self.theta_log,
            self.U1,
            self.U2,
            self.V1,
            self.V2,
            act_name,
            hc1_init,
            hc2_init,
            resets,
        )

        if self.batch_first:
            return y_btd_2h, (hc1, hc2)
        else:
            return y_btd_2h.transpose(0, 1), (hc1, hc2)


__all__ = ["LinearRTU_Triton"]
