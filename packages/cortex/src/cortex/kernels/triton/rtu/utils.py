from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


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


def hillis_steele_scan_inplace(
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


def hillis_steele_segmented_inplace(
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

    # Compose M and J
    g_new = g_r * g_l - p_r * p_l
    p_new = p_r * g_l + g_r * p_l
    jg_rl = jg_r * g_l - jp_r * p_l
    jp_rl = jp_r * g_l + jg_r * p_l
    jg_lr = g_r * jg_l - p_r * jp_l
    jp_lr = p_r * jg_l + g_r * jp_l
    jg_new = jg_rl + jg_lr
    jp_new = jp_rl + jp_lr

    # Compose Bc and Be
    bcx_rot = g_r * bcx_l - p_r * bcy_l
    bcy_rot = p_r * bcx_l + g_r * bcy_l
    bcx_new = bcx_rot + bcx_r
    bcy_new = bcy_rot + bcy_r
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


def scan_step_block_segmented(
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


__all__ = [
    "hillis_steele_scan_inplace",
    "hillis_steele_segmented_inplace",
    "scan_step_block_segmented",
]
