"""Triton-based, time-parallel RTU (low-rank input maps) with segmented resets.

Public API (functional): ``rtu_sequence_triton``
"""

from __future__ import annotations

from typing import Optional

import torch
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


@triton.jit
def _scan_step_block_segmented_kernel(
    G_ptr,
    P_ptr,
    JG_ptr,
    JP_ptr,
    BCx_ptr,
    BCy_ptr,
    BEx_ptr,
    BEy_ptr,
    Flags_ptr,
    T: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_t: tl.constexpr,
    offset: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Segmented Hillis–Steele step for the augmented state (hc, E).

    Each leaf encodes the affine map on S = [hc; E] with blocks:
      M = [[g,-p],[p,g]], J = [[jg,-jp],[jp,jg]]
      S' = [[M, 0],[J, M]] S + [Bc; Be]

    Composition across time obeys:
      M' = M_r M_l
      J' = J_r M_l + M_r J_l
      Bc' = M_r Bc_l + Bc_r
      Be' = M_r Be_l + J_r Bc_l + Be_r

    Segmented rule: if right is a head (flag=1), keep right element.
    """
    pid_lane = tl.program_id(0)
    pid_blk = tl.program_id(1)

    H = stride_b // stride_h
    b = pid_lane // H
    h = pid_lane % H

    t = offset + pid_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < T

    base = b * stride_b + h * stride_h
    ridx = base + t * stride_t

    g_r = tl.load(G_ptr + ridx, mask=mask, other=0.0)
    p_r = tl.load(P_ptr + ridx, mask=mask, other=0.0)
    jg_r = tl.load(JG_ptr + ridx, mask=mask, other=0.0)
    jp_r = tl.load(JP_ptr + ridx, mask=mask, other=0.0)
    bcx_r = tl.load(BCx_ptr + ridx, mask=mask, other=0.0)
    bcy_r = tl.load(BCy_ptr + ridx, mask=mask, other=0.0)
    bex_r = tl.load(BEx_ptr + ridx, mask=mask, other=0.0)
    bey_r = tl.load(BEy_ptr + ridx, mask=mask, other=0.0)
    fr = tl.load(Flags_ptr + ridx, mask=mask, other=1)

    lt = t - offset
    lidx = base + lt * stride_t
    g_l = tl.load(G_ptr + lidx, mask=mask, other=0.0)
    p_l = tl.load(P_ptr + lidx, mask=mask, other=0.0)
    jg_l = tl.load(JG_ptr + lidx, mask=mask, other=0.0)
    jp_l = tl.load(JP_ptr + lidx, mask=mask, other=0.0)
    bcx_l = tl.load(BCx_ptr + lidx, mask=mask, other=0.0)
    bcy_l = tl.load(BCy_ptr + lidx, mask=mask, other=0.0)
    bex_l = tl.load(BEx_ptr + lidx, mask=mask, other=0.0)
    bey_l = tl.load(BEy_ptr + lidx, mask=mask, other=0.0)
    fl = tl.load(Flags_ptr + lidx, mask=mask, other=1)

    # Compose M: Z' = Z_r * Z_l
    g_new = g_r * g_l - p_r * p_l
    p_new = p_r * g_l + g_r * p_l

    # Compose J: J' = J_r * Z_l + Z_r * J_l
    jg_rl = jg_r * g_l - jp_r * p_l
    jp_rl = jp_r * g_l + jg_r * p_l
    jg_lr = g_r * jg_l - p_r * jp_l
    jp_lr = p_r * jg_l + g_r * jp_l
    jg_new = jg_rl + jg_lr
    jp_new = jp_rl + jp_lr

    # Compose Bc: Bc' = Z_r * Bc_l + Bc_r
    bcx_rot = g_r * bcx_l - p_r * bcy_l
    bcy_rot = p_r * bcx_l + g_r * bcy_l
    bcx_new = bcx_rot + bcx_r
    bcy_new = bcy_rot + bcy_r

    # Compose Be: Be' = Z_r * Be_l + J_r * Bc_l + Be_r
    bex_rot = g_r * bex_l - p_r * bey_l
    bey_rot = p_r * bex_l + g_r * bey_l
    j_on_bc_x = jg_r * bcx_l - jp_r * bcy_l
    j_on_bc_y = jp_r * bcx_l + jg_r * bcy_l
    bex_new = bex_rot + j_on_bc_x + bex_r
    bey_new = bey_rot + j_on_bc_y + bey_r

    cond = fr == 0
    g_out = tl.where(cond, g_new, g_r)
    p_out = tl.where(cond, p_new, p_r)
    jg_out = tl.where(cond, jg_new, jg_r)
    jp_out = tl.where(cond, jp_new, jp_r)
    bcx_out = tl.where(cond, bcx_new, bcx_r)
    bcy_out = tl.where(cond, bcy_new, bcy_r)
    bex_out = tl.where(cond, bex_new, bex_r)
    bey_out = tl.where(cond, bey_new, bey_r)
    f_out = tl.minimum(fr + fl, 1)

    tl.store(G_ptr + ridx, g_out, mask=mask)
    tl.store(P_ptr + ridx, p_out, mask=mask)
    tl.store(JG_ptr + ridx, jg_out, mask=mask)
    tl.store(JP_ptr + ridx, jp_out, mask=mask)
    tl.store(BCx_ptr + ridx, bcx_out, mask=mask)
    tl.store(BCy_ptr + ridx, bcy_out, mask=mask)
    tl.store(BEx_ptr + ridx, bex_out, mask=mask)
    tl.store(BEy_ptr + ridx, bey_out, mask=mask)
    tl.store(Flags_ptr + ridx, f_out, mask=mask)


def _scan_step_block_segmented(
    g_bht: torch.Tensor,
    p_bht: torch.Tensor,
    jg_bht: torch.Tensor,
    jp_bht: torch.Tensor,
    bcx_bht: torch.Tensor,
    bcy_bht: torch.Tensor,
    bex_bht: torch.Tensor,
    bey_bht: torch.Tensor,
    flags_bht: torch.Tensor,
) -> None:
    """Host wrapper: segmented scan for the augmented block state."""
    assert g_bht.is_contiguous() and p_bht.is_contiguous()
    assert jg_bht.is_contiguous() and jp_bht.is_contiguous()
    assert bcx_bht.is_contiguous() and bcy_bht.is_contiguous()
    assert bex_bht.is_contiguous() and bey_bht.is_contiguous()
    assert flags_bht.is_contiguous()
    B, H, T = g_bht.shape
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
            _scan_step_block_segmented_kernel[grid](
                g_bht,
                p_bht,
                jg_bht,
                jp_bht,
                bcx_bht,
                bcy_bht,
                bex_bht,
                bey_bht,
                flags_bht,
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
        param_parallel: int = 1,
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
        ctx.param_parallel = bool(param_parallel)

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

        # Fully-parallel dynamics grads via block segmented scan over (hc, E)
        exp_nu_log = torch.exp(nu_log)
        exp_th_log = torch.exp(theta_log)
        # block-scan not used in closed-form path

        def block_scan_E(
            jg_param: torch.Tensor, jp_param: torch.Tensor, dgamma_param: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B_local, H_local, T_local = B, H, T
            # Per-time arrays for M (g,p) and J (jg,jp)
            g_bht = (
                g.view(1, H_local).expand(B_local, H_local).unsqueeze(-1).expand(B_local, H_local, T_local).contiguous()
            )
            p_bht = (
                phi.view(1, H_local)
                .expand(B_local, H_local)
                .unsqueeze(-1)
                .expand(B_local, H_local, T_local)
                .contiguous()
            )
            jg_bht = (
                jg_param.view(1, H_local)
                .expand(B_local, H_local)
                .unsqueeze(-1)
                .expand(B_local, H_local, T_local)
                .contiguous()
            )
            jp_bht = (
                jp_param.view(1, H_local)
                .expand(B_local, H_local)
                .unsqueeze(-1)
                .expand(B_local, H_local, T_local)
                .contiguous()
            )

            # Biases
            Bhc_x = (gamma.view(1, H_local, 1) * u1_bth.permute(0, 2, 1)).contiguous()
            Bhc_y = (gamma.view(1, H_local, 1) * u2_bth.permute(0, 2, 1)).contiguous()
            Be_x = (dgamma_param.view(1, H_local, 1) * u1_bth.permute(0, 2, 1)).contiguous()
            Be_y = (dgamma_param.view(1, H_local, 1) * u2_bth.permute(0, 2, 1)).contiguous()

            # Seed t=0 with contributions from c0 when not reset
            if T_local > 0:
                c0x = hc1_init_bh
                c0y = hc2_init_bh
                gg = g.view(1, H_local).expand(B_local, H_local)
                pp = phi.view(1, H_local).expand(B_local, H_local)
                rotx0 = gg * c0x - pp * c0y
                roty0 = pp * c0x + gg * c0y
                jx0 = (
                    jg_param.view(1, H_local).expand(B_local, H_local) * c0x
                    - jp_param.view(1, H_local).expand(B_local, H_local) * c0y
                )
                jy0 = (
                    jp_param.view(1, H_local).expand(B_local, H_local) * c0x
                    + jg_param.view(1, H_local).expand(B_local, H_local) * c0y
                )
                use_c0 = (~resets_bt[:, 0]).to(Bhc_x.dtype).view(B_local, 1)
                Bhc_x[:, :, 0] = Bhc_x[:, :, 0] + use_c0 * rotx0
                Bhc_y[:, :, 0] = Bhc_y[:, :, 0] + use_c0 * roty0
                Be_x[:, :, 0] = Be_x[:, :, 0] + use_c0 * jx0
                Be_y[:, :, 0] = Be_y[:, :, 0] + use_c0 * jy0

            flags = torch.zeros(B_local, H_local, T_local, dtype=torch.int32, device=x_btd.device)
            if T_local > 0:
                flags[:, :, 0] = 1
            if resets_bt.any():
                flags |= resets_bt.view(B_local, 1, T_local).expand(B_local, H_local, T_local).to(torch.int32)

            _scan_step_block_segmented(
                g_bht,
                p_bht,
                jg_bht,
                jp_bht,
                Bhc_x,
                Bhc_y,
                Be_x,
                Be_y,
                flags,
            )

            E1 = Be_x.permute(0, 2, 1).contiguous()
            E2 = Be_y.permute(0, 2, 1).contiguous()
            return E1, E2

        # Baseline-matching RTRL loop for nu/theta grads (keeps exact parity under resets)
        if ctx.param_parallel:
            # Fully-parallel for nu_log via closed-form; theta_log via short RTRL to meet tight tol
            c_bth2 = torch.stack([c1_bth, c2_bth], dim=-1)
            cprev_bth2 = torch.empty_like(c_bth2)
            c0_bh2 = torch.stack([hc1_init_bh, hc2_init_bh], dim=-1)
            cprev_bth2[:, 0, :, :] = c0_bh2
            if T > 1:
                cprev_bth2[:, 1:, :, :] = c_bth2[:, :-1, :, :]

            gate = (~resets_bt).to(dtype=torch.float64).unsqueeze(-1)
            lam1d = lam1_bth.double()
            lam2d = lam2_bth.double()
            cpx = cprev_bth2[..., 0].double()
            cpy = cprev_bth2[..., 1].double()

            dg = (gate * (lam1d * cpx + lam2d * cpy)).sum(dim=(0, 1))
            dphi = (gate * (-lam1d * cpy + lam2d * cpx)).sum(dim=(0, 1))
            dgamma = (lam1d * u1_bth.double() + lam2d * u2_bth.double()).sum(dim=(0, 1))

            exp_nu = torch.exp(nu_log.double())
            r_d = torch.exp(-torch.exp(nu_log.double()))
            gamma_d = torch.sqrt(torch.clamp(1.0 - r_d * r_d, min=0.0))
            g_d = g.double()
            phi_d = phi.double()

            dnu_log = (-exp_nu * (dg * g_d + dphi * phi_d) + exp_nu * (r_d * r_d / gamma_d) * dgamma).to(x_btd.dtype)

            # Theta via short RTRL loop (small BxH compute) for exactness
            exp_th_log = torch.exp(theta_log)
            d_g_d_th_bh = (-phi * exp_th_log).view(1, H)
            d_phi_d_th_bh = (g * exp_th_log).view(1, H)
            E_th_c1_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
            E_th_c2_bh = torch.zeros(B, H, device=x_btd.device, dtype=x_btd.dtype)
            grad_theta_log_h = torch.zeros_like(theta_log)
            hc1_prev = hc1_init_bh
            hc2_prev = hc2_init_bh
            g_bh = g.view(1, H)
            phi_bh = phi.view(1, H)
            for t in range(T):
                c1_t = c1_bth[:, t, :]
                c2_t = c2_bth[:, t, :]
                if resets_bt.any():
                    m = resets_bt[:, t].view(B, 1).to(dtype=hc1_prev.dtype)
                    one_minus = 1.0 - m
                    hc1_prev = hc1_prev * one_minus
                    hc2_prev = hc2_prev * one_minus
                    Eth_c1_old = E_th_c1_bh * one_minus
                    Eth_c2_old = E_th_c2_bh * one_minus
                else:
                    Eth_c1_old = E_th_c1_bh
                    Eth_c2_old = E_th_c2_bh
                E_th_c1_bh = d_g_d_th_bh * hc1_prev + g_bh * Eth_c1_old - d_phi_d_th_bh * hc2_prev - phi_bh * Eth_c2_old
                E_th_c2_bh = d_g_d_th_bh * hc2_prev + g_bh * Eth_c2_old + d_phi_d_th_bh * hc1_prev + phi_bh * Eth_c1_old
                grad_theta_log_h += torch.sum(eta1[:, t, :] * E_th_c1_bh + eta2[:, t, :] * E_th_c2_bh, dim=0)
                hc1_prev = c1_t
                hc2_prev = c2_t
            dth_log = grad_theta_log_h
        else:
            # Baseline-matching RTRL loop
            dL_dc1_bth = eta1
            dL_dc2_bth = eta2

            exp_nu_log = torch.exp(nu_log)
            exp_th_log = torch.exp(theta_log)
            r_loc = torch.exp(-torch.exp(nu_log))
            sqrt_1_minus_r2 = torch.sqrt(torch.clamp(1.0 - r_loc * r_loc, min=0.0))

            d_g_d_nu_bh = (-exp_nu_log * g).view(1, H)
            d_phi_d_nu_bh = (-exp_nu_log * phi).view(1, H)
            d_gamma_d_nu_bh = (exp_nu_log * r_loc * r_loc / sqrt_1_minus_r2).view(1, H)
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
                grad_theta_log_h += torch.sum(
                    dL_dc1_bth[:, t, :] * E_th_c1_bh + dL_dc2_bth[:, t, :] * E_th_c2_bh,
                    dim=0,
                )

                hc1_prev = c1_t
                hc2_prev = c2_t

            dnu_log = grad_nu_log_h
            dth_log = grad_theta_log_h

        # Initial-state gradients to enable cross-subsequence BPTT
        dhc1_init = lam1_bth[:, 0, :]
        dhc2_init = lam2_bth[:, 0, :]

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
            None,
        )


def rtu_sequence_triton(
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
    param_grads_parallel: bool = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Functional RTU interface (Triton autograd).

    Args are identical to the PyTorch functional API.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available. Please install `triton` and ensure CUDA is available.")
    y_btd_2h, hc1, hc2 = _LinearRTUFunctionLR_Triton.apply(
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
        int(param_grads_parallel),
    )
    return y_btd_2h, (hc1, hc2)


__all__ = ["rtu_sequence_triton"]
