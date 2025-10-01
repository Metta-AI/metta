"""Triton kernels for channel-mixing causal conv1d with per-timestep resets.

This module contains fused Triton implementations for channel-mixing (groups=1)
causal 1D convolution with support for per-timestep resets. Includes forward
and backward passes with fp32 accumulation for numerical stability.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _torch_dtype_to_triton(dtype: torch.dtype) -> object:
    """Map a torch dtype to its Triton counterpart for kernel compilation."""

    if dtype not in _TORCH_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {dtype} for causal conv1d Triton kernel")
    return _TORCH_TO_TRITON_DTYPE[dtype]


def _shared_memory_limit(device: torch.device) -> int:
    """Return the per-block shared memory limit for the given CUDA device."""

    props = torch.cuda.get_device_properties(device)
    shared_optin = getattr(props, "shared_memory_per_block_optin", 0)
    if shared_optin:
        return int(shared_optin)
    return int(props.shared_memory_per_block)


def _resolve_reduce_t(block_fi: int, block_fo: int, reduce_t: int, device: torch.device) -> int:
    """Choose a reduce_t that fits in shared memory by halving as needed."""

    if device.type != "cuda":
        return reduce_t

    shared_limit = _shared_memory_limit(device)
    current = reduce_t

    def required_bytes(rt: int) -> int:
        # Two tiles of size [rt, block] promoted to fp32 plus the fp32 accumulator tile.
        return 4 * (rt * (block_fo + block_fi) + block_fo * block_fi)

    while current > 1 and required_bytes(current) > shared_limit:
        current //= 2

    if required_bytes(current) > shared_limit:
        raise RuntimeError("Unable to choose reduce_t within shared memory budget for Triton kernel")

    return current


def _make_seg(resets: torch.Tensor) -> torch.Tensor:
    """Convert resets [B, T] to segment IDs via cumsum."""
    return torch.cumsum(resets.to(torch.int32), dim=1).contiguous()


@triton.jit
def _fwd_cm_kernel(
    X,
    W,
    BIAS,
    SEG,
    Y,
    B: tl.constexpr,
    T: tl.constexpr,
    FI: tl.constexpr,
    FO: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_FO: tl.constexpr,
    BLOCK_FI: tl.constexpr,
    N_FOB: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Forward kernel for channel-mixing causal conv1d with resets."""
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    b = pid0 // N_FOB
    fob = pid0 % N_FOB

    t_base = pid1 * BLOCK_T
    fo_base = fob * BLOCK_FO

    t_idx = t_base + tl.arange(0, BLOCK_T)
    fo_idx = fo_base + tl.arange(0, BLOCK_FO)

    mt = t_idx < T
    mfo = fo_idx < FO

    # Strides for [B, T, F]
    stride_x_b = T * FI
    stride_x_t = FI
    stride_x_f = 1
    stride_y_b = T * FO
    stride_y_t = FO
    stride_y_f = 1

    # Segment strides [B, T]
    stride_s_b = T
    stride_s_t = 1

    # Weight strides [FO, FI, KS]
    stride_w_fo = FI * KS
    stride_w_fi = KS
    stride_w_k = 1

    x_bt = X + b * stride_x_b
    y_bt = Y + b * stride_y_b
    seg_b = SEG + b * stride_s_b

    seg_t = tl.load(seg_b + t_idx * stride_s_t, mask=mt, other=0)

    acc = tl.zeros((BLOCK_T, BLOCK_FO), dtype=tl.float32)

    # Loop over kernel taps
    for k in range(KS):
        ts = t_idx - k
        k_rev = (KS - 1) - k
        mt_ts = (ts >= 0) & mt
        seg_src = tl.load(seg_b + ts * stride_s_t, mask=mt_ts, other=tl.full((), -1, tl.int32))
        mrow = mt_ts & (seg_t == seg_src)

        # Tile over input channels
        for fib in range(0, FI, BLOCK_FI):
            fi_idx = fib + tl.arange(0, BLOCK_FI)
            mfi = fi_idx < FI

            # X tile: [BLOCK_T, BLOCK_FI]
            x_ptrs = x_bt + ts[:, None] * stride_x_t + fi_idx[None, :] * stride_x_f
            x_tile = tl.load(x_ptrs, mask=(mrow[:, None] & mfi[None, :]), other=0.0)

            # W tile: [BLOCK_FO, BLOCK_FI] @k
            w_ptrs = W + fo_idx[:, None] * stride_w_fo + fi_idx[None, :] * stride_w_fi + k_rev * stride_w_k
            w_tile = tl.load(w_ptrs, mask=(mfo[:, None] & mfi[None, :]), other=0.0)
            w_tile_T = tl.trans(w_tile)

            acc += tl.dot(x_tile.to(tl.float32), w_tile_T.to(tl.float32))

    if HAS_BIAS:
        b_vals = tl.load(BIAS + fo_idx, mask=mfo, other=0.0)[None, :]
        acc += b_vals.to(tl.float32)

    y_ptrs = y_bt + t_idx[:, None] * stride_y_t + fo_idx[None, :] * stride_y_f
    tl.store(y_ptrs, acc, mask=(mt[:, None] & mfo[None, :]))


@triton.jit
def _bwd_cm_dx_kernel(
    GY,
    W,
    SEG,
    GX,
    B: tl.constexpr,
    T: tl.constexpr,
    FI: tl.constexpr,
    FO: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_FI: tl.constexpr,
    BLOCK_FO: tl.constexpr,
    N_FIB: tl.constexpr,
    DTYPE_GX: tl.constexpr,
):
    """Backward kernel for computing gradient w.r.t. input x."""
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    b = pid0 // N_FIB
    fib = pid0 % N_FIB

    t_base = pid1 * BLOCK_T
    fi_base = fib * BLOCK_FI

    t_idx = t_base + tl.arange(0, BLOCK_T)
    fi_idx = fi_base + tl.arange(0, BLOCK_FI)

    mt = t_idx < T
    mfi = fi_idx < FI

    # Strides
    stride_y_b = T * FO
    stride_y_t = FO
    stride_y_f = 1
    stride_x_b = T * FI
    stride_x_t = FI
    stride_x_f = 1

    stride_s_b = T
    stride_s_t = 1

    stride_w_fo = FI * KS
    stride_w_fi = KS
    stride_w_k = 1

    gy_bt = GY + b * stride_y_b
    gx_bt = GX + b * stride_x_b
    seg_b = SEG + b * stride_s_b

    seg_t0 = tl.load(seg_b + t_idx * stride_s_t, mask=mt, other=0)

    acc = tl.zeros((BLOCK_T, BLOCK_FI), dtype=tl.float32)

    for k in range(KS):
        tout = t_idx + k
        k_rev = (KS - 1) - k
        mt_out = (tout < T) & mt
        seg_out = tl.load(seg_b + tout * stride_s_t, mask=mt_out, other=tl.full((), -1, tl.int32))
        mrow = mt_out & (seg_out == seg_t0)

        # Reduce over FO in tiles
        for fob in range(0, FO, BLOCK_FO):
            fo_idx = fob + tl.arange(0, BLOCK_FO)
            mfo = fo_idx < FO

            # gy tile [T_blk, FO_blk]
            gy_ptrs = gy_bt + tout[:, None] * stride_y_t + fo_idx[None, :] * stride_y_f
            gy_tile = tl.load(gy_ptrs, mask=(mrow[:, None] & mfo[None, :]), other=0.0)

            # w tile [FO_blk, FI_blk] at k
            w_ptrs = W + fo_idx[:, None] * stride_w_fo + fi_idx[None, :] * stride_w_fi + k_rev * stride_w_k
            w_tile = tl.load(w_ptrs, mask=(mfo[:, None] & mfi[None, :]), other=0.0)

            acc += tl.dot(gy_tile.to(tl.float32), w_tile.to(tl.float32))

    gx_ptrs = gx_bt + t_idx[:, None] * stride_x_t + fi_idx[None, :] * stride_x_f
    tl.store(gx_ptrs, acc.to(DTYPE_GX), mask=(mt[:, None] & mfi[None, :]))


@triton.jit
def _bwd_cm_dw_kernel(
    X,
    GY,
    SEG,
    GW,
    B: tl.constexpr,
    T: tl.constexpr,
    FI: tl.constexpr,
    FO: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_FO: tl.constexpr,
    BLOCK_FI: tl.constexpr,
    REDUCE_T: tl.constexpr,
    N_FOB: tl.constexpr,
    N_FIB: tl.constexpr,
    DTYPE_GW: tl.constexpr,
):
    """Backward kernel for computing gradient w.r.t. weights."""
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    fob = pid0
    fib = pid1

    fo_base = fob * BLOCK_FO
    fi_base = fib * BLOCK_FI

    fo_idx = fo_base + tl.arange(0, BLOCK_FO)
    fi_idx = fi_base + tl.arange(0, BLOCK_FI)

    mfo = fo_idx < FO
    mfi = fi_idx < FI

    # Strides
    stride_x_b = T * FI
    stride_x_t = FI
    stride_x_f = 1
    stride_y_b = T * FO
    stride_y_t = FO
    stride_y_f = 1
    stride_s_b = T
    stride_s_t = 1
    stride_w_fo = FI * KS
    stride_w_fi = KS
    stride_w_k = 1

    # For each kernel tap k, accumulate one [FO_blk, FI_blk] tile across all B,T
    for k in range(KS):
        acc_k = tl.zeros((BLOCK_FO, BLOCK_FI), dtype=tl.float32)
        k_rev = (KS - 1) - k

        # Sweep batches and time in chunks
        for b in range(B):
            seg_b = SEG + b * stride_s_b
            x_b = X + b * stride_x_b
            gy_b = GY + b * stride_y_b

            for t0 in range(0, T, REDUCE_T):
                t_idx = t0 + tl.arange(0, REDUCE_T)
                mt = t_idx < T

                seg_t = tl.load(seg_b + t_idx * stride_s_t, mask=mt, other=0)

                ts = t_idx - k
                mts = (ts >= 0) & mt
                seg_src = tl.load(
                    seg_b + ts * stride_s_t,
                    mask=mts,
                    other=tl.full((), -1, tl.int32),
                )
                mrow = mts & (seg_t == seg_src)

                # gy tile: [RT, FO_blk]
                gy_ptrs = gy_b + t_idx[:, None] * stride_y_t + fo_idx[None, :] * stride_y_f
                gy_tile = tl.load(gy_ptrs, mask=(mrow[:, None] & mfo[None, :]), other=0.0).to(tl.float32)

                # x tile: [RT, FI_blk]
                x_ptrs = x_b + ts[:, None] * stride_x_t + fi_idx[None, :] * stride_x_f
                x_tile = tl.load(x_ptrs, mask=(mrow[:, None] & mfi[None, :]), other=0.0).to(tl.float32)

                # acc_k += gy.T @ x  -> [FO_blk, FI_blk]
                acc_k += tl.dot(tl.trans(gy_tile), x_tile)

        # Store GW slice for tap k
        gw_ptrs = GW + fo_idx[:, None] * stride_w_fo + fi_idx[None, :] * stride_w_fi + k_rev * stride_w_k
        tl.store(gw_ptrs, acc_k.to(DTYPE_GW), mask=(mfo[:, None] & mfi[None, :]))


@triton.jit
def _bwd_cm_bias_kernel(
    GY,
    GBIAS,
    B: tl.constexpr,
    T: tl.constexpr,
    FO: tl.constexpr,
    BLOCK_FO: tl.constexpr,
    REDUCE_T: tl.constexpr,
    DTYPE_GB: tl.constexpr,
):
    """Backward kernel for computing gradient w.r.t. bias."""
    pid = tl.program_id(axis=0)
    fo_base = pid * BLOCK_FO
    fo_idx = fo_base + tl.arange(0, BLOCK_FO)
    mfo = fo_idx < FO

    stride_y_b = T * FO
    stride_y_t = FO
    stride_y_f = 1

    acc = tl.zeros((BLOCK_FO,), dtype=tl.float32)

    for b in range(B):
        gy_b = GY + b * stride_y_b
        for t0 in range(0, T, REDUCE_T):
            t_idx = t0 + tl.arange(0, REDUCE_T)
            mt = t_idx < T
            gy_ptrs = gy_b + t_idx[:, None] * stride_y_t + fo_idx[None, :] * stride_y_f
            gy_tile = tl.load(gy_ptrs, mask=(mt[:, None] & mfo[None, :]), other=0.0)
            acc += tl.sum(gy_tile.to(tl.float32), axis=0)

    tl.store(GBIAS + fo_idx, acc.to(DTYPE_GB), mask=mfo)


class _ChannelMixCausalResetFn(torch.autograd.Function):
    """Autograd function for channel-mixing causal conv1d with resets."""

    @staticmethod
    def forward(
        ctx,
        state: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        resets: torch.Tensor,
        block_t: int,
        block_fi: int,
        block_fo: int,
        reduce_t: int,
    ):
        assert state.is_cuda and x.is_cuda and w.is_cuda and resets.is_cuda
        assert state.is_contiguous(), "state must be contiguous [B,KS,FI]"
        assert x.is_contiguous(), "x must be contiguous [B,T,FI]"
        assert w.is_contiguous(), "w must be contiguous [FO,FI,KS]"
        if bias is not None:
            assert bias.is_cuda and bias.is_contiguous()
        B, KS, FI_state = state.shape
        Bx, T, FI = x.shape
        assert B == Bx, "Batch mismatch between state and x"
        assert FI_state == FI, "Feature mismatch between state and x"
        FO, FIw, KS = w.shape
        assert FI == FIw, "FI mismatch"

        resets = resets.contiguous()
        state_prefix = state.contiguous()
        x_full = torch.cat([state_prefix, x], dim=1)
        resets_prefix = torch.zeros((B, KS), dtype=resets.dtype, device=resets.device)
        resets_full = torch.cat([resets_prefix, resets], dim=1)
        seg = _make_seg(resets_full)

        T_full = T + KS

        y_full = torch.empty((B, T_full, FO), device=x.device, dtype=x.dtype)

        N_FOB = triton.cdiv(FO, block_fo)
        grid = (B * N_FOB, triton.cdiv(T_full, block_t))
        HAS_BIAS = bias is not None
        bias_ptr = bias if HAS_BIAS else x

        _fwd_cm_kernel[grid](
            x_full,
            w,
            bias_ptr,
            seg,
            y_full,
            B,
            T_full,
            FI,
            FO,
            KS,
            block_t,
            block_fo,
            block_fi,
            N_FOB,
            HAS_BIAS=HAS_BIAS,
        )

        y = y_full[:, KS:, :]

        ctx.save_for_backward(
            state_prefix,
            x,
            w,
            seg,
            (bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)),
        )
        ctx.has_bias = HAS_BIAS
        ctx.block_t = block_t
        ctx.block_fi = block_fi
        ctx.block_fo = block_fo
        ctx.reduce_t = reduce_t
        ctx.KS = KS
        ctx.T = T
        return y

    @staticmethod
    def backward(ctx, gy: torch.Tensor):
        state, x, w, seg, _bias_saved = ctx.saved_tensors
        B, KS, FI = state.shape
        _, T, _ = x.shape
        FO, FIw, KS = w.shape
        assert FI == FIw
        HAS_BIAS = ctx.has_bias
        bt = ctx.block_t
        bfi = ctx.block_fi
        bfo = ctx.block_fo
        rt = ctx.reduce_t
        T_full = T + KS

        gx_dtype = _torch_dtype_to_triton(x.dtype)
        gw_dtype = _torch_dtype_to_triton(w.dtype)
        gy_dtype = _torch_dtype_to_triton(gy.dtype)

        rt_effective = _resolve_reduce_t(bfi, bfo, rt, x.device)

        x_full = torch.cat([state, x], dim=1)
        gy_full = torch.cat([torch.zeros((B, KS, FO), device=gy.device, dtype=gy.dtype), gy], dim=1)

        # grad x
        gx_full = torch.empty_like(x_full)
        N_FIB = triton.cdiv(FI, bfi)
        grid_dx = (B * N_FIB, triton.cdiv(T_full, bt))
        _bwd_cm_dx_kernel[grid_dx](
            gy_full,
            w,
            seg,
            gx_full,
            B,
            T_full,
            FI,
            FO,
            KS,
            bt,
            bfi,
            bfo,
            N_FIB,
            DTYPE_GX=gx_dtype,
        )
        gx_state = gx_full[:, :KS, :]
        gx = gx_full[:, KS:, :]

        # grad w
        gw = torch.empty_like(w)
        N_FOB = triton.cdiv(FO, bfo)
        N_FIB = triton.cdiv(FI, bfi)
        grid_dw = (N_FOB, N_FIB)
        _bwd_cm_dw_kernel[grid_dw](
            x_full,
            gy_full,
            seg,
            gw,
            B,
            T_full,
            FI,
            FO,
            KS,
            bfo,
            bfi,
            rt_effective,
            N_FOB,
            N_FIB,
            DTYPE_GW=gw_dtype,
        )

        # grad bias
        if HAS_BIAS:
            gb = torch.empty((FO,), device=x.device, dtype=gy.dtype)
            grid_b = (triton.cdiv(FO, bfo),)
            _bwd_cm_bias_kernel[grid_b](
                gy_full,
                gb,
                B,
                T_full,
                FO,
                bfo,
                rt_effective,
                DTYPE_GB=gy_dtype,
            )
        else:
            gb = None

        return gx_state, gx, gw, gb, None, None, None, None, None


def channelmix_causal_conv1d_with_resets_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    resets: torch.Tensor,
    *,
    block_t: int = 128,
    block_fi: int = 64,
    block_fo: int = 64,
    reduce_t: int = 256,
) -> torch.Tensor:
    """Fused channel-mixing (groups=1) causal Conv1d with per-timestep resets.

    Uses Triton for efficient computation with fp32 accumulation.

    Args:
        x: Input tensor [B, T, FI], CUDA, contiguous
        w: Weight tensor [FO, FI, KS], CUDA, contiguous
        bias: Optional bias [FO] or None
        resets: Reset mask [B, T], CUDA
        block_t: Time block size (multiple of 16 recommended)
        block_fi: Input feature block size
        block_fo: Output feature block size
        reduce_t: Reduction block size for gradients

    Returns:
        Output tensor [B, T, FO]
    """
    return _ChannelMixCausalResetFn.apply(state, x, w, bias, resets, block_t, block_fi, block_fo, reduce_t)


__all__ = ["channelmix_causal_conv1d_with_resets_triton"]
