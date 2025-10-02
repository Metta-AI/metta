"""Triton LSTM kernel with per-timestep reset support.

This module adapts the open-source ``flashrnn`` kernels from Maximilian Beck
to provide a single-layer LSTM forward/backward implementation that matches the
PyTorch reference used by :class:`cortex.cells.lstm.LSTMCell`. The kernels have
been extended to honour batch-first reset masks, which zero the recurrent state
before a step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import triton.language as tl
from einops import rearrange
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

import triton
from triton import OutOfResources

# -----------------------------------------------------------------------------
# Utility helpers (mirrors flashrnn.triton_fused.triton_utils)


_TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# Batch tile size shared between forward/backward kernels. We operate on small
# tiles to support narrow batches while keeping the per-thread work manageable.
_BATCH_TILE_SIZE = 16
_MATMUL_K_TILE = 16


def _torch_to_triton_dtype(dtype: torch.dtype) -> tl.dtype:
    if dtype not in _TORCH_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {dtype} for Triton kernel")
    return _TORCH_TO_TRITON_DTYPE[dtype]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_multiple_of(n: int, multiple_of: int) -> int:
    return ((n + multiple_of - 1) // multiple_of) * multiple_of


def _dtype_registry_key(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported dtype for Triton LSTM: {dtype}")


# -----------------------------------------------------------------------------
# Forward kernel (adapted from flashrnn.triton_fused.lstm_fw)


@triton.jit
def _triton_tanh(x):
    return (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))


@triton.autotune(
    configs=[triton.Config({"siz_B": _BATCH_TILE_SIZE}, num_stages=1, num_warps=4)],
    key=["siz_B", "T", "B", "NH", "DH"],
)
@triton.jit
def _lstm_forward_kernel(
    states_initial,  # (NH, NS, B, DH)
    Wx,  # (NH, T, NGI, B, DH)
    R,  # (NH, NGR, DH_in, DH_out)
    b,  # (NH, NGI, DH)
    resets,  # (NH, T, B) or dummy tensor when HAS_RESETS is False
    states_all,  # (NH, T + 1, NS, B, DH)
    gates_all,  # (NH, T, NGI, B, DH)
    # Dimensions (compile-time constants)
    T: tl.constexpr,
    NS: tl.constexpr,
    B: tl.constexpr,
    B_TRUE,
    NH: tl.constexpr,
    DH: tl.constexpr,
    NGI: tl.constexpr,
    NGR: tl.constexpr,
    siz_B: tl.constexpr,
    MATMUL_K_TILE: tl.constexpr,
    OUTPUT_GATES: tl.constexpr,
    HAS_RESETS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    idx_head = tl.program_id(0)
    idx_batch = tl.program_id(1)

    # Strides (batch dimension is contiguous for all tensors)
    stride_states_all_head = (T + 1) * NS * B * DH
    stride_states_all_time = NS * B * DH
    stride_states_initial_head = NS * B * DH
    stride_Wx_head = T * NGI * B * DH
    stride_Wx_time = NGI * B * DH
    stride_gates_head = T * NGI * B * DH
    stride_gates_time = NGI * B * DH

    batch_offsets = idx_batch * siz_B + tl.arange(0, siz_B)
    b_true = tl.full((siz_B,), B_TRUE, dtype=batch_offsets.dtype)
    off_valid = batch_offsets < b_true

    # Load initial hidden and cell states (float32 accumulator for stability)
    h_ptr = tl.make_block_ptr(
        base=states_initial + idx_head * stride_states_initial_head,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    h_t = tl.load(h_ptr, boundary_check=(0, 1))
    h_t = tl.where(off_valid[:, None], h_t, 0.0)
    h_t = h_t.to(tl.float32)

    c_ptr = tl.make_block_ptr(
        base=states_initial + idx_head * stride_states_initial_head + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    c_t = tl.load(c_ptr, boundary_check=(0, 1))
    c_t = tl.where(off_valid[:, None], c_t, 0.0)
    c_t = c_t.to(tl.float32)

    # Store initial state (index 0)
    h_store_ptr = tl.make_block_ptr(
        base=states_all + idx_head * stride_states_all_head,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(h_store_ptr, h_t.to(DTYPE), boundary_check=(0, 1))

    c_store_ptr = tl.make_block_ptr(
        base=states_all + idx_head * stride_states_all_head + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(c_store_ptr, c_t.to(DTYPE), boundary_check=(0, 1))

    # Load recurrent weights and biases once per kernel launch
    base_R = R + idx_head * NGR * DH * DH
    ptr_R_i = tl.make_block_ptr(
        base=base_R + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_f = tl.make_block_ptr(
        base=base_R + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_z = tl.make_block_ptr(
        base=base_R + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_o = tl.make_block_ptr(
        base=base_R + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )

    R_i = tl.load(ptr_R_i).to(tl.float32)
    R_f = tl.load(ptr_R_f).to(tl.float32)
    R_z = tl.load(ptr_R_z).to(tl.float32)
    R_o = tl.load(ptr_R_o).to(tl.float32)
    ptr_R_i = tl.make_block_ptr(
        base=base_R + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_f = tl.make_block_ptr(
        base=base_R + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_z = tl.make_block_ptr(
        base=base_R + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_o = tl.make_block_ptr(
        base=base_R + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )

    R_i = tl.load(ptr_R_i).to(tl.float32)
    R_f = tl.load(ptr_R_f).to(tl.float32)
    R_z = tl.load(ptr_R_z).to(tl.float32)
    R_o = tl.load(ptr_R_o).to(tl.float32)
    ptr_R_i = tl.make_block_ptr(
        base=base_R + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_f = tl.make_block_ptr(
        base=base_R + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_z = tl.make_block_ptr(
        base=base_R + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_o = tl.make_block_ptr(
        base=base_R + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )

    R_i = tl.load(ptr_R_i).to(tl.float32)
    R_f = tl.load(ptr_R_f).to(tl.float32)
    R_z = tl.load(ptr_R_z).to(tl.float32)
    R_o = tl.load(ptr_R_o).to(tl.float32)
    R_i_base = base_R + 0 * DH * DH
    R_f_base = base_R + 1 * DH * DH
    R_z_base = base_R + 2 * DH * DH
    R_o_base = base_R + 3 * DH * DH

    use_tf32 = DTYPE == tl.float32

    bias_i = tl.load(b + idx_head * NGI * DH + 0 * DH + tl.arange(0, DH)).to(tl.float32)
    bias_f = tl.load(b + idx_head * NGI * DH + 1 * DH + tl.arange(0, DH)).to(tl.float32)
    bias_z = tl.load(b + idx_head * NGI * DH + 2 * DH + tl.arange(0, DH)).to(tl.float32)
    bias_o = tl.load(b + idx_head * NGI * DH + 3 * DH + tl.arange(0, DH)).to(tl.float32)

    for t in range(T):
        # Optional reset mask for this time step and batch block
        if HAS_RESETS:
            reset_vals = tl.load(
                resets + idx_head * T * B + t * B + idx_batch * siz_B + tl.arange(0, siz_B),
                mask=off_valid,
                other=0.0,
            )
            reset_vals = reset_vals.to(tl.float32).reshape((siz_B, 1))
            keep = 1.0 - reset_vals
            h_t = h_t * keep
            c_t = c_t * keep

            prev_h_ptr = tl.make_block_ptr(
                base=states_all + idx_head * stride_states_all_head + t * stride_states_all_time,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            tl.store(prev_h_ptr, h_t.to(DTYPE), boundary_check=(0, 1))

            prev_c_ptr = tl.make_block_ptr(
                base=states_all + idx_head * stride_states_all_head + t * stride_states_all_time + 1 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            tl.store(prev_c_ptr, c_t.to(DTYPE), boundary_check=(0, 1))

        # Load input contributions (i, f, z, o)
        gate_base = Wx + idx_head * stride_Wx_head + t * stride_Wx_time

        ptr_ix = tl.make_block_ptr(
            base=gate_base + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_fx = tl.make_block_ptr(
            base=gate_base + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_zx = tl.make_block_ptr(
            base=gate_base + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_ox = tl.make_block_ptr(
            base=gate_base + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )

        Ix = tl.load(ptr_ix, boundary_check=(0, 1))
        Fx = tl.load(ptr_fx, boundary_check=(0, 1))
        Zx = tl.load(ptr_zx, boundary_check=(0, 1))
        Ox = tl.load(ptr_ox, boundary_check=(0, 1))

        Ix = tl.where(off_valid[:, None], Ix, 0.0).to(tl.float32)
        Fx = tl.where(off_valid[:, None], Fx, 0.0).to(tl.float32)
        Zx = tl.where(off_valid[:, None], Zx, 0.0).to(tl.float32)
        Ox = tl.where(off_valid[:, None], Ox, 0.0).to(tl.float32)

        Rh_i = tl.zeros((siz_B, DH), dtype=tl.float32)
        Rh_f = tl.zeros((siz_B, DH), dtype=tl.float32)
        Rh_z = tl.zeros((siz_B, DH), dtype=tl.float32)
        Rh_o = tl.zeros((siz_B, DH), dtype=tl.float32)

        for k in tl.static_range(0, DH, MATMUL_K_TILE):
            h_block_ptr = tl.make_block_ptr(
                base=states_all + idx_head * stride_states_all_head + t * stride_states_all_time,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, k),
                block_shape=(siz_B, MATMUL_K_TILE),
                order=(0, 1),
            )
            h_block = tl.load(h_block_ptr, boundary_check=(0, 1))
            h_block = tl.where(off_valid[:, None], h_block, 0.0).to(tl.float32)

            ptr_Ri_tile = tl.make_block_ptr(
                base=R_i_base,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(k, 0),
                block_shape=(MATMUL_K_TILE, DH),
                order=(0, 1),
            )
            ptr_Rf_tile = tl.make_block_ptr(
                base=R_f_base,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(k, 0),
                block_shape=(MATMUL_K_TILE, DH),
                order=(0, 1),
            )
            ptr_Rz_tile = tl.make_block_ptr(
                base=R_z_base,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(k, 0),
                block_shape=(MATMUL_K_TILE, DH),
                order=(0, 1),
            )
            ptr_Ro_tile = tl.make_block_ptr(
                base=R_o_base,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(k, 0),
                block_shape=(MATMUL_K_TILE, DH),
                order=(0, 1),
            )

            R_i_tile = tl.load(ptr_Ri_tile, boundary_check=(0, 1)).to(tl.float32)
            R_f_tile = tl.load(ptr_Rf_tile, boundary_check=(0, 1)).to(tl.float32)
            R_z_tile = tl.load(ptr_Rz_tile, boundary_check=(0, 1)).to(tl.float32)
            R_o_tile = tl.load(ptr_Ro_tile, boundary_check=(0, 1)).to(tl.float32)

            Rh_i += tl.dot(h_block, R_i_tile, allow_tf32=use_tf32)
            Rh_f += tl.dot(h_block, R_f_tile, allow_tf32=use_tf32)
            Rh_z += tl.dot(h_block, R_z_tile, allow_tf32=use_tf32)
            Rh_o += tl.dot(h_block, R_o_tile, allow_tf32=use_tf32)

        # Gate pre-activations + activations
        ibar = Ix + Rh_i + bias_i[None, :]
        fbar = Fx + Rh_f + bias_f[None, :]
        zbar = Zx + Rh_z + bias_z[None, :]
        obar = Ox + Rh_o + bias_o[None, :]

        gate_i_act = tl.sigmoid(ibar)
        gate_f_act = tl.sigmoid(fbar)
        gate_z_act = _triton_tanh(zbar)
        gate_o_act = tl.sigmoid(obar)

        c_next = gate_f_act * c_t + gate_i_act * gate_z_act
        h_next = gate_o_act * _triton_tanh(c_next)

        c_next = tl.where(off_valid[:, None], c_next, 0.0)
        h_next = tl.where(off_valid[:, None], h_next, 0.0)
        gate_i_act = tl.where(off_valid[:, None], gate_i_act, 0.0)
        gate_f_act = tl.where(off_valid[:, None], gate_f_act, 0.0)
        gate_z_act = tl.where(off_valid[:, None], gate_z_act, 0.0)
        gate_o_act = tl.where(off_valid[:, None], gate_o_act, 0.0)

        # Store next states (time t+1)
        h_next_ptr = tl.make_block_ptr(
            base=states_all + idx_head * stride_states_all_head + (t + 1) * stride_states_all_time,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(h_next_ptr, h_next.to(DTYPE), boundary_check=(0, 1))

        c_next_ptr = tl.make_block_ptr(
            base=states_all + idx_head * stride_states_all_head + (t + 1) * stride_states_all_time + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(c_next_ptr, c_next.to(DTYPE), boundary_check=(0, 1))

        if OUTPUT_GATES:
            gate_store_base = gates_all + idx_head * stride_gates_head + t * stride_gates_time

            ptr_gi = tl.make_block_ptr(
                base=gate_store_base + 0 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            ptr_gf = tl.make_block_ptr(
                base=gate_store_base + 1 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            ptr_gz = tl.make_block_ptr(
                base=gate_store_base + 2 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            ptr_go = tl.make_block_ptr(
                base=gate_store_base + 3 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_batch * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )

            tl.store(ptr_gi, gate_i_act.to(DTYPE), boundary_check=(0, 1))
            tl.store(ptr_gf, gate_f_act.to(DTYPE), boundary_check=(0, 1))
            tl.store(ptr_gz, gate_z_act.to(DTYPE), boundary_check=(0, 1))
            tl.store(ptr_go, gate_o_act.to(DTYPE), boundary_check=(0, 1))

        h_t = h_next
        c_t = c_next


# -----------------------------------------------------------------------------
# Backward kernel (adapted from flashrnn.triton_fused.lstm_bw)


@triton.jit
def _lstm_backward_kernel(
    delta_states_all_outside,  # (NH, T, NS, B, DH)
    delta_states_last_outside,  # (NH, NS, B, DH)
    R,  # (NH, NGR, DH_out, DH_in)
    states_all,  # (NH, T + 1, NS, B, DH)
    gates_all,  # (NH, T, NGI, B, DH)
    resets,  # (NH, T, B)
    delta_states_initial,  # (NH, NS, B, DH)
    delta_Wx,  # (NH, T, NGI, B, DH)
    delta_R,  # (num_blocks_B, NH, NGR, DH_out, DH_in)
    delta_b,  # (num_blocks_B, NH, NGI, DH)
    T: tl.constexpr,
    NS: tl.constexpr,
    B: tl.constexpr,
    B_TRUE,
    NH: tl.constexpr,
    DH: tl.constexpr,
    NGI: tl.constexpr,
    NGR: tl.constexpr,
    siz_B: tl.constexpr,
    HAS_RESETS: tl.constexpr,
    DTYPE: tl.constexpr,
    backward_recurrent_clip_val: tl.constexpr,
):
    idx_head = tl.program_id(0)
    idx_batch = tl.program_id(1)

    stride_states_all_head = (T + 1) * NS * B * DH
    stride_states_all_time = NS * B * DH
    stride_gates_head = T * NGI * B * DH
    stride_gates_time = NGI * B * DH
    stride_delta_states_head = T * NS * B * DH
    stride_delta_states_time = NS * B * DH

    batch_offsets = idx_batch * siz_B + tl.arange(0, siz_B)
    b_true = tl.full((siz_B,), B_TRUE, dtype=batch_offsets.dtype)
    off_valid = batch_offsets < b_true

    delta_h_ptr = tl.make_block_ptr(
        base=delta_states_last_outside + idx_head * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    delta_c_ptr = tl.make_block_ptr(
        base=delta_states_last_outside + idx_head * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )

    delta_h_next = tl.load(delta_h_ptr, boundary_check=(0, 1))
    delta_h_next = tl.where(off_valid[:, None], delta_h_next, 0.0).to(tl.float32)
    delta_c_next = tl.load(delta_c_ptr, boundary_check=(0, 1))
    delta_c_next = tl.where(off_valid[:, None], delta_c_next, 0.0).to(tl.float32)

    base_R = R + idx_head * NGR * DH * DH
    ptr_R_i = tl.make_block_ptr(
        base=base_R + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_f = tl.make_block_ptr(
        base=base_R + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_z = tl.make_block_ptr(
        base=base_R + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_R_o = tl.make_block_ptr(
        base=base_R + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )

    R_i = tl.load(ptr_R_i).to(tl.float32)
    R_f = tl.load(ptr_R_f).to(tl.float32)
    R_z = tl.load(ptr_R_z).to(tl.float32)
    R_o = tl.load(ptr_R_o).to(tl.float32)

    grad_R_i = tl.zeros((DH, DH), dtype=tl.float32)
    grad_R_f = tl.zeros((DH, DH), dtype=tl.float32)
    grad_R_z = tl.zeros((DH, DH), dtype=tl.float32)
    grad_R_o = tl.zeros((DH, DH), dtype=tl.float32)

    grad_b_i = tl.zeros((DH,), dtype=tl.float32)
    grad_b_f = tl.zeros((DH,), dtype=tl.float32)
    grad_b_z = tl.zeros((DH,), dtype=tl.float32)
    grad_b_o = tl.zeros((DH,), dtype=tl.float32)

    for t in range(T - 1, -1, -1):
        # Gate activations from forward pass
        gate_base = gates_all + idx_head * stride_gates_head + t * stride_gates_time
        ptr_gi = tl.make_block_ptr(
            base=gate_base + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_gf = tl.make_block_ptr(
            base=gate_base + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_gz = tl.make_block_ptr(
            base=gate_base + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_go = tl.make_block_ptr(
            base=gate_base + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )

        gate_i = tl.load(ptr_gi, boundary_check=(0, 1))
        gate_f = tl.load(ptr_gf, boundary_check=(0, 1))
        gate_z = tl.load(ptr_gz, boundary_check=(0, 1))
        gate_o = tl.load(ptr_go, boundary_check=(0, 1))

        gate_i = tl.where(off_valid[:, None], gate_i, 0.0).to(tl.float32)
        gate_f = tl.where(off_valid[:, None], gate_f, 0.0).to(tl.float32)
        gate_z = tl.where(off_valid[:, None], gate_z, 0.0).to(tl.float32)
        gate_o = tl.where(off_valid[:, None], gate_o, 0.0).to(tl.float32)

        ptr_c_t = tl.make_block_ptr(
            base=states_all + idx_head * stride_states_all_head + (t + 1) * stride_states_all_time + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_c_prev = tl.make_block_ptr(
            base=states_all + idx_head * stride_states_all_head + t * stride_states_all_time + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_h_prev = tl.make_block_ptr(
            base=states_all + idx_head * stride_states_all_head + t * stride_states_all_time + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )

        c_t = tl.load(ptr_c_t, boundary_check=(0, 1))
        c_t = tl.where(off_valid[:, None], c_t, 0.0).to(tl.float32)
        c_prev = tl.load(ptr_c_prev, boundary_check=(0, 1))
        c_prev = tl.where(off_valid[:, None], c_prev, 0.0).to(tl.float32)
        h_prev = tl.load(ptr_h_prev, boundary_check=(0, 1))
        h_prev = tl.where(off_valid[:, None], h_prev, 0.0).to(tl.float32)

        delta_h_out = tl.make_block_ptr(
            base=delta_states_all_outside + idx_head * stride_delta_states_head + t * stride_delta_states_time,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        delta_h_from_outside = tl.load(delta_h_out, boundary_check=(0, 1))
        delta_h_from_outside = tl.where(off_valid[:, None], delta_h_from_outside, 0.0).to(tl.float32)

        delta_c_out = tl.make_block_ptr(
            base=delta_states_all_outside
            + idx_head * stride_delta_states_head
            + t * stride_delta_states_time
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        delta_c_from_outside = tl.load(delta_c_out, boundary_check=(0, 1))
        delta_c_from_outside = tl.where(off_valid[:, None], delta_c_from_outside, 0.0).to(tl.float32)

        delta_h_total = delta_h_from_outside + delta_h_next
        delta_c_total = delta_c_from_outside + delta_c_next

        tanh_c = _triton_tanh(c_t)
        delta_c_total = delta_c_total + delta_h_total * gate_o * (1.0 - tanh_c * tanh_c)

        delta_i = delta_c_total * gate_z * gate_i * (1.0 - gate_i)
        delta_f = delta_c_total * c_prev * gate_f * (1.0 - gate_f)
        delta_z = delta_c_total * gate_i * (1.0 - gate_z * gate_z)
        delta_o = delta_h_total * tanh_c * gate_o * (1.0 - gate_o)

        delta_c_next = delta_c_total * gate_f

        delta_i_f32 = delta_i.to(tl.float32)
        delta_f_f32 = delta_f.to(tl.float32)
        delta_z_f32 = delta_z.to(tl.float32)
        delta_o_f32 = delta_o.to(tl.float32)

        delta_h_prev = tl.sum(delta_i_f32[:, :, None] * R_i[None, :, :], axis=1)
        delta_h_prev += tl.sum(delta_f_f32[:, :, None] * R_f[None, :, :], axis=1)
        delta_h_prev += tl.sum(delta_z_f32[:, :, None] * R_z[None, :, :], axis=1)
        delta_h_prev += tl.sum(delta_o_f32[:, :, None] * R_o[None, :, :], axis=1)

        if backward_recurrent_clip_val > 0:
            clip_val = backward_recurrent_clip_val
            delta_h_prev = tl.maximum(tl.minimum(delta_h_prev, clip_val), -clip_val)

        if HAS_RESETS:
            reset_vals = tl.load(
                resets + idx_head * T * B + t * B + idx_batch * siz_B + tl.arange(0, siz_B),
                mask=off_valid,
                other=0.0,
            )
            keep = (1.0 - reset_vals.to(tl.float32)).reshape((siz_B, 1))
            delta_h_prev = delta_h_prev * keep
            delta_c_next = delta_c_next * keep

        h_prev_f32 = h_prev.to(tl.float32)
        grad_R_i += tl.sum(delta_i_f32[:, :, None] * h_prev_f32[:, None, :], axis=0)
        grad_R_f += tl.sum(delta_f_f32[:, :, None] * h_prev_f32[:, None, :], axis=0)
        grad_R_z += tl.sum(delta_z_f32[:, :, None] * h_prev_f32[:, None, :], axis=0)
        grad_R_o += tl.sum(delta_o_f32[:, :, None] * h_prev_f32[:, None, :], axis=0)

        grad_b_i += tl.sum(delta_i, axis=0)
        grad_b_f += tl.sum(delta_f, axis=0)
        grad_b_z += tl.sum(delta_z, axis=0)
        grad_b_o += tl.sum(delta_o, axis=0)

        delta_base = delta_Wx + idx_head * T * NGI * B * DH + t * NGI * B * DH
        ptr_di = tl.make_block_ptr(
            base=delta_base + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_df = tl.make_block_ptr(
            base=delta_base + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_dz = tl.make_block_ptr(
            base=delta_base + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        ptr_do = tl.make_block_ptr(
            base=delta_base + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_batch * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )

        tl.store(ptr_di, delta_i.to(DTYPE), boundary_check=(0, 1))
        tl.store(ptr_df, delta_f.to(DTYPE), boundary_check=(0, 1))
        tl.store(ptr_dz, delta_z.to(DTYPE), boundary_check=(0, 1))
        tl.store(ptr_do, delta_o.to(DTYPE), boundary_check=(0, 1))

        delta_h_next = delta_h_prev

    # store gradients for initial states
    ptr_dh0 = tl.make_block_ptr(
        base=delta_states_initial + idx_head * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    ptr_dc0 = tl.make_block_ptr(
        base=delta_states_initial + idx_head * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_batch * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(ptr_dh0, delta_h_next.to(DTYPE), boundary_check=(0, 1))
    tl.store(ptr_dc0, delta_c_next.to(DTYPE), boundary_check=(0, 1))

    block_base = idx_batch * NH * NGR * DH * DH + idx_head * NGR * DH * DH
    ptr_dR_i = tl.make_block_ptr(
        base=delta_R + block_base + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_dR_f = tl.make_block_ptr(
        base=delta_R + block_base + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_dR_z = tl.make_block_ptr(
        base=delta_R + block_base + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    ptr_dR_o = tl.make_block_ptr(
        base=delta_R + block_base + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )

    tl.store(ptr_dR_i, grad_R_i.to(DTYPE))
    tl.store(ptr_dR_f, grad_R_f.to(DTYPE))
    tl.store(ptr_dR_z, grad_R_z.to(DTYPE))
    tl.store(ptr_dR_o, grad_R_o.to(DTYPE))

    base_b = idx_batch * NH * NGI * DH + idx_head * NGI * DH
    tl.store(delta_b + base_b + 0 * DH + tl.arange(0, DH), grad_b_i.to(DTYPE))
    tl.store(delta_b + base_b + 1 * DH + tl.arange(0, DH), grad_b_f.to(DTYPE))
    tl.store(delta_b + base_b + 2 * DH + tl.arange(0, DH), grad_b_z.to(DTYPE))
    tl.store(delta_b + base_b + 3 * DH + tl.arange(0, DH), grad_b_o.to(DTYPE))


# -----------------------------------------------------------------------------
# Host-side helpers wrapping Triton kernels


@dataclass
class _LSTMKernelInputs:
    states_initial: Tensor  # (NS, B_eff, NH, DH)
    Wx: Tensor  # (B_eff, T, NGI, NH, DH)
    R: Tensor  # (NGR, NH, DH_out, DH_in)
    b: Tensor  # (NGI, NH, DH)
    resets: Tensor | None  # (B_eff, T)
    true_batch: int


def _prepare_padded_inputs(
    states_initial: Tensor,
    Wx: Tensor,
    R: Tensor,
    b: Tensor,
    resets: Tensor | None,
) -> _LSTMKernelInputs:
    NS, B, NH, DH = states_initial.shape
    _, T, NGI, _, _ = Wx.shape
    assert _is_power_of_two(DH), "Hidden size must be a power of two for Triton kernel"

    MIN_BATCH = _BATCH_TILE_SIZE
    effective_B = _next_multiple_of(B, MIN_BATCH)

    if effective_B != B:
        pad = effective_B - B
        pad_states = torch.zeros(NS, pad, NH, DH, dtype=states_initial.dtype, device=states_initial.device)
        states_initial = torch.cat([states_initial, pad_states], dim=1)

        pad_inputs = torch.zeros(pad, T, NGI, NH, DH, dtype=Wx.dtype, device=Wx.device)
        Wx = torch.cat([Wx, pad_inputs], dim=0)

        if resets is not None:
            pad_resets = torch.zeros(pad, T, dtype=resets.dtype, device=resets.device)
            resets = torch.cat([resets, pad_resets], dim=0)

    return _LSTMKernelInputs(states_initial, Wx, R, b, resets, true_batch=B)


def _lstm_forward(
    *,
    states_initial: Tensor,
    Wx: Tensor,
    R: Tensor,
    b: Tensor,
    reset_mask: Tensor | None,
    output_gates_and_states_initial: bool,
) -> tuple[tuple[Tensor, Tensor], Tensor | None]:
    """Run the Triton forward kernel and optionally collect gates."""

    inputs = _prepare_padded_inputs(states_initial, Wx, R, b, reset_mask)
    NS, B_eff, NH, DH = inputs.states_initial.shape
    _, T, NGI, _, _ = inputs.Wx.shape

    states_k = rearrange(inputs.states_initial, "ns b nh dh -> nh ns b dh").contiguous()
    Wx_k = rearrange(inputs.Wx, "b t ngi nh dh -> nh t ngi b dh").contiguous()
    R_k = rearrange(R, "ngr nh dhout dhin -> nh ngr dhin dhout").contiguous()
    b_k = rearrange(b, "ngi nh dh -> nh ngi dh").contiguous()

    if inputs.resets is not None:
        resets_k = rearrange(inputs.resets, "b t -> 1 t b").contiguous()
    else:
        resets_k = torch.empty(1, 1, 1, device=states_initial.device, dtype=states_initial.dtype)

    states_all = torch.empty((NH, T + 1, NS, B_eff, DH), device=states_initial.device, dtype=states_initial.dtype)
    gates_all = (
        torch.empty((NH, T, NGI, B_eff, DH), device=states_initial.device, dtype=states_initial.dtype)
        if output_gates_and_states_initial
        else None
    )

    def _grid(meta):
        siz_B = meta["siz_B"]
        if siz_B < _BATCH_TILE_SIZE:
            raise OutOfResources(required=siz_B, limit=_BATCH_TILE_SIZE, name="siz_B")
        if siz_B > B_eff:
            raise OutOfResources(required=siz_B, limit=B_eff, name="siz_B")
        return (NH, triton.cdiv(B_eff, siz_B))

    _lstm_forward_kernel[_grid](
        states_initial=states_k,
        Wx=Wx_k,
        R=R_k,
        b=b_k,
        resets=resets_k,
        states_all=states_all,
        gates_all=gates_all if gates_all is not None else states_all,
        T=T,
        NS=NS,
        B=B_eff,
        B_TRUE=inputs.true_batch,
        NH=NH,
        DH=DH,
        NGI=NGI,
        NGR=NGI,
        OUTPUT_GATES=output_gates_and_states_initial,
        HAS_RESETS=reset_mask is not None,
        DTYPE=_torch_to_triton_dtype(states_initial.dtype),
        MATMUL_K_TILE=_MATMUL_K_TILE,
    )

    states_out = rearrange(states_all, "nh t ns b dh -> t ns b nh dh")
    if not output_gates_and_states_initial:
        states_out = states_out[:, :, : inputs.true_batch, :, :]
        last_state = states_out[-1]
        return (states_out[1:], last_state), None

    gates_out = rearrange(gates_all, "nh t ngi b dh -> t ngi b nh dh")

    states_pre = states_out[:-1]
    c_prev = states_pre[:, 1, :, :, :]
    gate_i = gates_out[:, 0, :, :, :]
    gate_f = gates_out[:, 1, :, :, :]
    gate_z = gates_out[:, 2, :, :, :]
    gate_o = gates_out[:, 3, :, :, :]

    c_next = gate_f * c_prev + gate_i * gate_z
    h_next = gate_o * torch.tanh(c_next)
    states_next = torch.stack((h_next, c_next), dim=1)

    return (states_out, states_out[-1], states_next), gates_out


def _lstm_backward(
    *,
    delta_states_all_outside: Tensor,
    delta_states_last_outside: Tensor,
    R: Tensor,
    states_all: Tensor,
    gates_all: Tensor,
    reset_mask: Tensor | None,
    true_batch: int,
    backward_recurrent_clip_val: float | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = R.dtype
    device = R.device
    T, NS, B_true, NH, DH = delta_states_all_outside.shape
    _, _, B_eff, _, _ = states_all.shape

    if B_true != B_eff:
        pad = B_eff - B_true
        pad_all = torch.zeros((T, NS, pad, NH, DH), dtype=dtype, device=device)
        delta_states_all_outside = torch.cat([delta_states_all_outside, pad_all], dim=2)

        pad_last = torch.zeros((NS, pad, NH, DH), dtype=dtype, device=device)
        delta_states_last_outside = torch.cat([delta_states_last_outside, pad_last], dim=1)

    if reset_mask is not None:
        resets_eff = reset_mask
        if resets_eff.shape[0] != B_eff:
            pad = B_eff - resets_eff.shape[0]
            resets_eff = torch.cat(
                [resets_eff, torch.zeros(pad, resets_eff.shape[1], dtype=reset_mask.dtype, device=device)],
                dim=0,
            )
        resets_k = rearrange(resets_eff, "b t -> 1 t b").contiguous()
    else:
        resets_k = torch.empty(1, 1, 1, device=device, dtype=dtype)

    R_k = rearrange(R, "ngr nh dhout dhin -> nh ngr dhout dhin").contiguous()
    delta_all_k = rearrange(delta_states_all_outside, "t ns b nh dh -> nh t ns b dh").contiguous()
    delta_last_k = rearrange(delta_states_last_outside, "ns b nh dh -> nh ns b dh").contiguous()
    states_all_k = rearrange(states_all, "t ns b nh dh -> nh t ns b dh").contiguous()
    gates_all_k = rearrange(gates_all, "t ngi b nh dh -> nh t ngi b dh").contiguous()

    siz_B = _BATCH_TILE_SIZE
    num_blocks = triton.cdiv(B_eff, siz_B)
    delta_states_initial = torch.empty((NH, NS, B_eff, DH), dtype=dtype, device=device)
    delta_Wx = torch.empty((NH, T, 4, B_eff, DH), dtype=dtype, device=device)
    delta_R = torch.empty((num_blocks, NH, 4, DH, DH), dtype=dtype, device=device)
    delta_b = torch.empty((num_blocks, NH, 4, DH), dtype=dtype, device=device)

    clip_val = backward_recurrent_clip_val if backward_recurrent_clip_val is not None else -1.0

    _lstm_backward_kernel[(NH, num_blocks)](
        delta_states_all_outside=delta_all_k,
        delta_states_last_outside=delta_last_k,
        R=R_k,
        states_all=states_all_k,
        gates_all=gates_all_k,
        resets=resets_k,
        delta_states_initial=delta_states_initial,
        delta_Wx=delta_Wx,
        delta_R=delta_R,
        delta_b=delta_b,
        T=T,
        NS=NS,
        B=B_eff,
        B_TRUE=true_batch,
        NH=NH,
        DH=DH,
        NGI=4,
        NGR=4,
        siz_B=siz_B,
        HAS_RESETS=reset_mask is not None,
        DTYPE=_torch_to_triton_dtype(dtype),
        backward_recurrent_clip_val=clip_val,
        num_warps=4,
        num_stages=1,
    )

    delta_states_initial = rearrange(delta_states_initial, "nh ns b dh -> ns b nh dh")[:, :true_batch, :, :]
    delta_Wx = rearrange(delta_Wx, "nh t ngi b dh -> b t ngi nh dh")[:true_batch]
    delta_R = delta_R.sum(0)
    delta_R = rearrange(delta_R, "nh ngr dhout dhin -> ngr nh dhout dhin")
    delta_b = delta_b.sum(0)
    delta_b = rearrange(delta_b, "nh ngi dh -> ngi nh dh")

    return delta_states_initial, delta_Wx, delta_R, delta_b


# -----------------------------------------------------------------------------
# Autograd wrapper (mirrors flashrnn.triton_fused.fwbw with reset support)


def _rnn_fwbw_generator(autocast_kernel_dtype: torch.dtype) -> torch.autograd.Function:
    forward_seq = _lstm_forward
    backward_seq = _lstm_backward

    class _LSTMTRFunction(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        def forward(
            ctx,
            states_initial: Tensor,
            Wx: Tensor,
            R: Tensor,
            b: Tensor,
            reset_mask: Tensor | None,
            backward_recurrent_clip_val: float | None = None,
        ) -> tuple[Tensor, Tensor]:
            (states_all, last_state, states_next), gates_all = forward_seq(
                states_initial=states_initial,
                Wx=Wx,
                R=R,
                b=b,
                reset_mask=reset_mask,
                output_gates_and_states_initial=True,
            )

            ctx.save_for_backward(
                states_all,
                gates_all,
                R,
                torch.tensor(
                    [-1.0 if backward_recurrent_clip_val is None else backward_recurrent_clip_val],
                    device=Wx.device,
                    dtype=Wx.dtype,
                ),
                reset_mask if reset_mask is not None else torch.empty(0, device=Wx.device, dtype=Wx.dtype),
            )
            ctx.true_batch = Wx.shape[0]

            if last_state.ndim == 4:
                last_state_out = last_state[:, : ctx.true_batch, ...]
            else:
                last_state_out = last_state[:, :, : ctx.true_batch, ...]

            seq_trim = states_next.narrow(2, 0, ctx.true_batch).clone()
            return seq_trim, last_state_out

        @staticmethod
        @custom_bwd(device_type="cuda")
        def backward(
            ctx,
            delta_states_all_outside: Tensor,
            delta_states_last_outside: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, None]:
            (states_all, gates_all, R, clip_tensor, reset_tensor) = ctx.saved_tensors
            clip_val = clip_tensor.item()
            true_batch = ctx.true_batch

            reset_mask = None
            if reset_tensor.numel() > 0:
                T = delta_states_all_outside.shape[0]
                reset_mask = reset_tensor.reshape(true_batch, T)

            delta_states_initial, delta_Wx, delta_R, delta_b = backward_seq(
                delta_states_all_outside=delta_states_all_outside,
                delta_states_last_outside=delta_states_last_outside,
                R=R,
                states_all=states_all,
                gates_all=gates_all,
                reset_mask=reset_mask,
                true_batch=true_batch,
                backward_recurrent_clip_val=None if clip_val < 0 else clip_val,
            )
            return delta_states_initial, delta_Wx, delta_R, delta_b, None, None

    return _LSTMTRFunction


_LSTM_FUNC_REGISTRY = {
    "float32": _rnn_fwbw_generator(torch.float32),
    "float16": _rnn_fwbw_generator(torch.float16),
    "bfloat16": _rnn_fwbw_generator(torch.bfloat16),
}


def _lstm_triton_autograd(
    *,
    states_initial: Tensor,
    Wx: Tensor,
    R: Tensor,
    b: Tensor,
    reset_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    registry_key = _dtype_registry_key(Wx.dtype)
    func = _LSTM_FUNC_REGISTRY[registry_key]
    return func.apply(states_initial, Wx, R, b, reset_mask)


# -----------------------------------------------------------------------------
# Public wrapper used by the LSTM cell


def lstm_sequence_triton(
    *,
    lstm: nn.LSTM,
    x_seq: Tensor,
    h0_bf: Tensor,
    c0_bf: Tensor,
    resets: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the Triton LSTM kernel for a batch-first sequence.

    Args:
        lstm: Single-layer ``nn.LSTM`` module without projection.
        x_seq: Input of shape ``[B, T, H]``.
        h0_bf: Initial hidden state ``[B, 1, H]``.
        c0_bf: Initial cell state ``[B, 1, H]``.
        resets: Optional reset mask ``[B, T]`` (truthy -> reset before timestep).

    Returns:
        Tuple ``(y_seq, hn_bf, cn_bf)`` matching the PyTorch implementation.
    """

    if not x_seq.is_cuda:
        raise RuntimeError("Triton LSTM requires CUDA tensors")
    if lstm.num_layers != 1:
        raise RuntimeError("Triton LSTM only supports num_layers == 1")
    if lstm.proj_size != 0:
        raise RuntimeError("Triton LSTM does not support projected outputs")

    hidden_size = lstm.hidden_size
    if not _is_power_of_two(hidden_size):
        raise RuntimeError("Hidden size must be a power of two for Triton backend")

    weight_ih = lstm.weight_ih_l0
    weight_hh = lstm.weight_hh_l0
    bias_ih = lstm.bias_ih_l0 if lstm.bias else None
    bias_hh = lstm.bias_hh_l0 if lstm.bias else None

    bias = None
    if bias_ih is not None and bias_hh is not None:
        bias = bias_ih + bias_hh
    elif bias_ih is not None:
        bias = bias_ih
    elif bias_hh is not None:
        bias = bias_hh
    else:
        bias = torch.zeros_like(weight_ih[:, 0])

    B, T, _ = x_seq.shape
    dtype = x_seq.dtype

    h0 = h0_bf.squeeze(1)
    c0 = c0_bf.squeeze(1)

    states_initial = torch.stack((h0, c0), dim=0).unsqueeze(2)  # (NS=2, B, NH=1, H)

    Wx = torch.nn.functional.linear(x_seq, weight_ih, bias=None)
    Wx = Wx.view(B, T, 4, 1, hidden_size)

    R = weight_hh.view(4, 1, hidden_size, hidden_size)
    b_vec = bias.view(4, 1, hidden_size)

    reset_mask = resets.to(dtype) if resets is not None else None

    all_states, last_state = _lstm_triton_autograd(
        states_initial=states_initial,
        Wx=Wx,
        R=R,
        b=b_vec,
        reset_mask=reset_mask,
    )

    h_seq = all_states[:, 0, :, 0, :]
    y_seq = h_seq.permute(1, 0, 2).contiguous()
    hn = last_state[0, :, 0, :]
    cn = last_state[1, :, 0, :]

    return y_seq, hn.unsqueeze(1), cn.unsqueeze(1)


__all__ = ["lstm_sequence_triton"]
