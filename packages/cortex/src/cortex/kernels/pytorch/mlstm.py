"""mLSTM backend implementations.

This module contains the core backend functions for the mLSTM cell,
including parallel and recurrent processing modes, as well as the
chunkwise implementation for efficient sequence processing.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch


def mlstm_recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    reset_mask: Optional[torch.Tensor] = None,  # B - boolean mask for resets
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state (torch.Tensor): (B, NH, DH, DH)
        n_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, 1, DH)
        k (torch.Tensor): (B, NH, 1, DH)
        v (torch.Tensor): (B, NH, 1, DH)
        igate_preact (torch.Tensor): (B, NH, 1, 1)
        fgate_preact (torch.Tensor): (B, NH, 1, 1)

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            hidden_state: (B, NH, DH)
            new_states: (
                c_state_new: (B, NH, DH, DH),
                n_state_new: (B, NH, DH, 1),
                m_state_new: (B, NH, 1, 1),
            )
    """
    B, NH, S, DH = q.shape

    # Apply reset mask if provided - reset states before processing
    if reset_mask is not None:
        _dtype = c_state.dtype
        mask_expanded = reset_mask.to(dtype=_dtype).view(B, 1, 1, 1)
        c_state = c_state * (1.0 - mask_expanded)
        n_state = n_state * (1.0 - mask_expanded)
        m_state = m_state * (1.0 - mask_expanded)

    # projections (avoid in-place ops on views to keep autograd happy)
    q = q.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)
    k = k.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)
    v = v.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)

    # gates
    log_fg_act = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, 1, 1)

    # update rule
    m_state_new = torch.max(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    k_scaled = k / math.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (k_scaled @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    h_num = q.transpose(-1, -2) @ c_state_new  # (B, NH, 1, DH)

    qn_dotproduct = q.transpose(-1, -2) @ n_state_new  # (B, NH, 1, 1)
    max_val = torch.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


def mlstm_chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,  # B, NH, S, DH
    values: torch.Tensor,  # B, NH, S, DH
    igate_preact: torch.Tensor,  # B, NH, S
    fgate_preact: torch.Tensor,  # B, NH, S
    initial_C: Optional[torch.Tensor] = None,  # B, NH, DH, DH
    initial_n: Optional[torch.Tensor] = None,  # B, NH, DH or (B, NH, DH, 1)
    initial_m: Optional[torch.Tensor] = None,  # B, NH, 1, 1
    reset_mask: Optional[torch.Tensor] = None,  # B, S - boolean mask for resets
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Chunkwise mLSTM implementation that also returns the final states.

    Notes
    - Works for any sequence length by padding to a multiple of ``chunk_size``.
    - Ensures returned ``n`` state has shape ``(B, NH, DH, 1)`` for consistency
      with the recurrent step backend and the public API expectations.
    """
    B, NH, S_orig, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # When the sequence length fits into a single chunk, compute outputs and
    # final states using the recurrent step kernel for exact step semantics.
    if S_orig <= chunk_size:
        c = initial_C if initial_C is not None else torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
        n = initial_n if initial_n is not None else torch.zeros((B, NH, DH, 1), dtype=_dtype, device=_device)
        if n.dim() == 3:
            n = n.unsqueeze(-1)
        m = initial_m if initial_m is not None else torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device)

        outs = []
        # Single-step function expects shapes (B, NH, 1, DH) and gates (B, NH, 1, 1)
        for s in range(S_orig):
            # Apply reset mask if provided - reset states before processing timestep s
            if reset_mask is not None:
                # reset_mask is [B, S], get mask for current timestep
                mask_s = reset_mask[:, s].to(dtype=_dtype).view(B, 1, 1, 1)
                c = c * (1.0 - mask_s)
                n = n * (1.0 - mask_s)
                m = m * (1.0 - mask_s)

            q_s = queries[:, :, s : s + 1, :]
            k_s = keys[:, :, s : s + 1, :]
            v_s = values[:, :, s : s + 1, :]
            i_s = igate_preact[:, :, s : s + 1].unsqueeze(-1)
            f_s = fgate_preact[:, :, s : s + 1].unsqueeze(-1)

            h_step, (c, n, m) = mlstm_recurrent_step_stabilized_simple(
                c_state=c,
                n_state=n,
                m_state=m,
                q=q_s,
                k=k_s,
                v=v_s,
                igate_preact=i_s,
                fgate_preact=f_s,
                eps=eps,
            )
            outs.append(h_step)

        h_seq = torch.cat(outs, dim=2)  # (B, NH, S, DH)
        if return_last_state:
            return h_seq, (c, n, m)
        else:
            return h_seq

    # Determine chunking layout without padding; handle tail via recurrent step to keep states exact
    CS = chunk_size
    NS_full = S_orig // CS
    S_main = NS_full * CS
    R = S_orig - S_main

    # Initialize boundary states
    C = None
    n = None
    m = None
    outputs = []

    # Build reset-aware masks for the main chunked region (vectorized, no fallback)
    if reset_mask is not None and NS_full > 0:
        # Reshape reset mask to chunk view and broadcast over heads
        reset_main = reset_mask[:, :S_main].view(B, 1, NS_full, CS)  # (B, 1, NS, CS)
        reset_main = reset_main.expand(B, NH, NS_full, CS)  # (B, NH, NS, CS)

        # Integer prefix sums along time within chunk
        reset_int = reset_main.to(dtype=torch.int64)
        prefix_inclusive = torch.cumsum(reset_int, dim=-1)  # (#resets up to and including t)
        prefix_exclusive = prefix_inclusive - reset_int  # (#resets strictly before t)

        # Mask: for output at time t, inter-chunk contributions vanish if any reset occurred in [0..t]
        no_reset_prefix = prefix_inclusive.eq(0)  # (B, NH, NS, CS) bool

        # Mask: for pair (t, s) within chunk, contributions vanish if any reset in (s..t]
        # Using inclusive counts ensures that a reset at t masks all s < t while keeping s == t valid.
        same_segment = prefix_inclusive.unsqueeze(-1).eq(prefix_inclusive.unsqueeze(-2))  # (B, NH, NS, CS, CS)

        # Mask: for end-of-chunk aggregation, only timesteps with no reset after them survive to the end
        total_resets = prefix_inclusive[..., -1:]  # (B, NH, NS, 1)
        has_reset_after = (total_resets - prefix_inclusive).gt(0)  # (B, NH, NS, CS)
        survive_to_end = ~has_reset_after  # (B, NH, NS, CS)
    else:
        reset_main = None
        no_reset_prefix = None
        same_segment = None
        survive_to_end = None

    # No within-chunk resets, proceed with chunk processing
    # Form full-size chunks for the main segment
    if NS_full > 0:
        q_main = queries[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)  # no scaling
        k_main = keys[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)
        v_main = values[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)
        # Scale keys (aligns with step/parallel backends)
        k_main_scaled = k_main / math.sqrt(DH)
        i_main = igate_preact[:, :, :S_main].view(B, NH, NS_full, CS)
        f_main = fgate_preact[:, :, :S_main].view(B, NH, NS_full, CS)

    if NS_full > 0:
        # Forget gates over time within each chunk (main part)
        log_fgates = torch.nn.functional.logsigmoid(f_main)
        log_fgates_acc = log_fgates.cumsum(dim=3)

        loggates = (i_main - log_fgates_acc)[..., None]  # (B, NH, NS, CS, 1)
        # Apply reset-aware masking for end-of-chunk aggregation
        if survive_to_end is not None:
            lg_full = loggates + log_fgates_acc[:, :, :, -1, None, None]
            lg_full = lg_full.masked_fill(~survive_to_end[..., None], float("-inf"))
            m_loc, _ = torch.max(lg_full, dim=3, keepdim=True)
            loggates = lg_full - m_loc
        else:
            m_loc, _ = torch.max(loggates + log_fgates_acc[:, :, :, -1, None, None], dim=3, keepdim=True)
            loggates = loggates + log_fgates_acc[:, :, :, -1, None, None] - m_loc

        kv = k_main_scaled.transpose(-1, -2) @ (v_main * loggates.exp())
        ksum = (k_main_scaled * loggates.exp()).sum(dim=-2)

        C = torch.zeros((B, NH, NS_full + 1, DH, DH), device=_device, dtype=_dtype)
        n = torch.zeros((B, NH, NS_full + 1, DH), device=_device, dtype=_dtype)
        if initial_C is not None:
            C[:, :, 0] = initial_C
        if initial_n is not None:
            n[:, :, 0] = initial_n.squeeze(-1) if (initial_n.dim() == 4) else initial_n

        m = torch.zeros((B, NH, NS_full + 1, 1, 1), device=_device, dtype=_dtype)
        if initial_m is not None:
            m[:, :, 0] = initial_m

        for i in range(1, NS_full + 1):
            # Check for resets at chunk boundary (start of chunk i)
            if reset_mask is not None:
                chunk_start_idx = (i - 1) * CS
                # Check if any reset occurs at the start of this chunk
                reset_at_boundary = reset_mask[:, chunk_start_idx].to(dtype=_dtype).view(B, 1, 1, 1)
                # Apply reset to states before processing
                C[:, :, i - 1] = C[:, :, i - 1] * (1.0 - reset_at_boundary)
                n[:, :, i - 1] = n[:, :, i - 1] * (1.0 - reset_at_boundary[:, :, :, 0])
                m[:, :, i - 1] = m[:, :, i - 1] * (1.0 - reset_at_boundary)

            m[:, :, i] = torch.maximum(
                log_fgates_acc[:, :, i - 1, -1, None, None] + m[:, :, i - 1],
                m_loc[:, :, i - 1],
            )
            C[:, :, i] = (
                C[:, :, i - 1].clone()
                * (log_fgates_acc[:, :, i - 1, -1, None, None] + m[:, :, i - 1] - m[:, :, i]).exp()
                + kv[:, :, i - 1] * (m_loc[:, :, i - 1] - m[:, :, i]).exp()
            )
            n[:, :, i] = (
                n[:, :, i - 1].clone()
                * (log_fgates_acc[:, :, i - 1, -1, None] + m[:, :, i - 1, 0] - m[:, :, i, 0]).exp()
                + ksum[:, :, i - 1] * (m_loc[:, :, i - 1, 0] - m[:, :, i, 0]).exp()
            )

        # Within-chunk decay and combination (main part)
        log_fgates_rep = log_fgates_acc[:, :, :, :, None].repeat(1, 1, 1, 1, CS)
        log_fg_matrix = (
            log_fgates_rep
            - log_fgates_rep.transpose(-1, -2)
            - torch.triu(float("inf") * torch.ones([1, 1, 1, CS, CS], device=_device, dtype=_dtype), diagonal=1)
        )

        log_D_matrix = log_fg_matrix + i_main[:, :, :, :, None].transpose(-2, -1)
        # Zero out pairwise contributions that cross a reset inside the chunk
        if same_segment is not None:
            log_D_matrix = log_D_matrix.masked_fill(~same_segment, float("-inf"))
        D_max, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)

        stab = torch.maximum(D_max, m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])
        inter_factor = (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
        # Mask inter-chunk contributions after any reset within the chunk prefix
        if no_reset_prefix is not None:
            inter_factor = inter_factor * no_reset_prefix[..., None]
        inter_C = (q_main * inter_factor) @ C[:, :, :-1]
        inter_n = (q_main * inter_factor) @ n[:, :, :-1, :, None]

        log_D_matrix_stabilized = log_D_matrix - stab
        D_matrix = torch.exp(log_D_matrix_stabilized)

        qk_matrix = q_main @ k_main_scaled.transpose(-2, -1)
        E_matrix = qk_matrix * D_matrix

        normalizer = torch.maximum((E_matrix.sum(dim=-1, keepdim=True) + inter_n).abs(), torch.exp(-stab))
        E_matrix_normalized = E_matrix / (normalizer + eps)

        intra = E_matrix_normalized @ v_main
        inter = inter_C / (normalizer + eps)
        output_main = (intra + inter).view((B, NH, S_main, DH))
        outputs.append(output_main)

    # Handle remainder via recurrent stepping for exactness
    if R > 0:
        # Initialize states if no main part processed
        if C is None:
            C_last = initial_C if initial_C is not None else torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
            n_last = initial_n if initial_n is not None else torch.zeros((B, NH, DH, 1), dtype=_dtype, device=_device)
            if n_last.dim() == 3:
                n_last = n_last.unsqueeze(-1)
            m_last = initial_m if initial_m is not None else torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device)
        else:
            C_last = C[:, :, -1]
            n_last = n[:, :, -1].unsqueeze(-1)
            m_last = m[:, :, -1]

        outs_rem = []
        for s in range(S_main, S_orig):
            # Apply reset mask if provided - reset states before processing timestep s
            if reset_mask is not None:
                mask_s = reset_mask[:, s].to(dtype=_dtype).view(B, 1, 1, 1)
                C_last = C_last * (1.0 - mask_s)
                n_last = n_last * (1.0 - mask_s)
                m_last = m_last * (1.0 - mask_s)

            q_s = queries[:, :, s : s + 1, :]
            k_s = keys[:, :, s : s + 1, :]
            v_s = values[:, :, s : s + 1, :]
            i_s = igate_preact[:, :, s : s + 1].unsqueeze(-1)
            f_s = fgate_preact[:, :, s : s + 1].unsqueeze(-1)

            h_step, (C_last, n_last, m_last) = mlstm_recurrent_step_stabilized_simple(
                c_state=C_last,
                n_state=n_last,
                m_state=m_last,
                q=q_s,
                k=k_s,
                v=v_s,
                igate_preact=i_s,
                fgate_preact=f_s,
                eps=eps,
            )
            outs_rem.append(h_step)

        outputs.append(torch.cat(outs_rem, dim=2))

    output = torch.cat(outputs, dim=2) if outputs else torch.empty(B, NH, 0, DH, device=_device, dtype=_dtype)

    if return_last_state:
        if R > 0:
            return output, (C_last, n_last, m_last)
        else:
            # NS_full >= 1 here
            return output, (C[:, :, -1], n[:, :, -1].unsqueeze(-1), m[:, :, -1])
    else:
        return output


def mlstm_chunkwise_triton(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    initial_C: Optional[torch.Tensor] = None,
    initial_n: Optional[torch.Tensor] = None,
    initial_m: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,  # B, S - boolean mask for resets
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Triton-accelerated chunkwise mLSTM implementation.

    Uses optimized Triton kernels when available via lazy import.
    Falls back to simple implementation if triton is not available.

    Args:
        queries: (B, NH, S, DH)
        keys: (B, NH, S, DH)
        values: (B, NH, S, DH)
        igate_preact: (B, NH, S)
        fgate_preact: (B, NH, S)
        initial_C: (B, NH, DH, DH), optional
        initial_n: (B, NH, DH) or (B, NH, DH, 1), optional
        initial_m: (B, NH, 1, 1), optional
        chunk_size: Size of chunks for processing
        return_last_state: Whether to return final states
        eps: Small constant for numerical stability

    Returns:
        Output tensor (B, NH, S, DH) and optionally final states (C, n, m)
    """
    # Lazy import to avoid loading unnecessary modules
    try:
        from cortex.kernels.triton.mlstm.torch import mlstm_chunkwise__xl_chunk
    except ImportError:
        # Fallback if triton kernels are not available
        return mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            initial_C=initial_C,
            initial_n=initial_n,
            initial_m=initial_m,
            reset_mask=reset_mask,
            chunk_size=chunk_size,
            return_last_state=return_last_state,
            eps=eps,
            **kwargs,
        )

    # Use Triton kernel in a single call; internal kernels handle resets and chunking
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    c_initial = initial_C if initial_C is not None else torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_initial = initial_n if initial_n is not None else torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    if n_initial.dim() == 4:
        n_initial = n_initial.squeeze(-1)
    m_initial = initial_m if initial_m is not None else torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    if m_initial.dim() == 4:
        m_initial = m_initial.squeeze(-1)

    # Transparent padding to multiple of 16 with identity steps
    S_orig = S
    pad = (16 - (S_orig % 16)) % 16
    if pad > 0:
        zeros_t = torch.zeros(B, NH, pad, DH, dtype=_dtype, device=_device)
        q_pad = torch.cat([queries, zeros_t], dim=2)
        k_pad = torch.cat([keys, zeros_t], dim=2)
        v_pad = torch.cat([values, zeros_t], dim=2)

        # For padded timesteps: forget≈1, input≈0 so state is unchanged
        if _dtype in (torch.float16, torch.bfloat16):
            pos = torch.tensor(10.0, dtype=_dtype, device=_device)
        else:
            pos = torch.tensor(20.0, dtype=_dtype, device=_device)
        neg = -pos
        i_tail = neg.expand(B, NH, pad)
        f_tail = pos.expand(B, NH, pad)
        i_pad = torch.cat([igate_preact, i_tail], dim=2)
        f_pad = torch.cat([fgate_preact, f_tail], dim=2)

        rm_pad = None
        if reset_mask is not None:
            rm_tail = torch.zeros(B, pad, dtype=reset_mask.dtype, device=_device)
            rm_pad = torch.cat([reset_mask, rm_tail], dim=1)

        result = mlstm_chunkwise__xl_chunk(
            q=q_pad,
            k=k_pad,
            v=v_pad,
            i=i_pad,
            f=f_pad,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_state,
            eps=eps,
            chunk_size=chunk_size,
            reset_mask=rm_pad,
        )

        if return_last_state:
            h_pad, (c_last, n_last, m_last) = result
            h_seq = h_pad[:, :, :S_orig, :]
        else:
            h_pad = result
            h_seq = h_pad[:, :, :S_orig, :]
    else:
        result = mlstm_chunkwise__xl_chunk(
            q=queries,
            k=keys,
            v=values,
            i=igate_preact,
            f=fgate_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_state,
            eps=eps,
            chunk_size=chunk_size,
            reset_mask=reset_mask,
        )

        if return_last_state:
            h_seq, (c_last, n_last, m_last) = result
        else:
            h_seq = result

    if return_last_state:
        if n_last.dim() == 3:
            n_last = n_last.unsqueeze(-1)
        if m_last.dim() == 3:
            m_last = m_last.unsqueeze(-1)
        scale = 1.0 / math.sqrt(DH)
        return h_seq, (c_last * scale, n_last * scale, m_last)
    else:
        return h_seq


__all__ = [
    "mlstm_chunkwise_simple",
    "mlstm_chunkwise_triton",
    "mlstm_recurrent_step_stabilized_simple",
]
