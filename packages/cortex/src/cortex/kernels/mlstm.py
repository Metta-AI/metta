"""mLSTM backend implementations.

This module contains the core backend functions for the mLSTM cell,
including parallel and recurrent processing modes, as well as the
chunkwise implementation for efficient sequence processing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# Check if triton is available
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def mlstm_parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


def mlstm_recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
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

    # Form full-size chunks for the main segment
    if NS_full > 0:
        q_main = queries[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)  # no scaling
        k_main = keys[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)
        v_main = values[:, :, :S_main, :].view(B, NH, NS_full, CS, DH)
        # Scale keys (aligns with step/parallel backends)
        k_main_scaled = k_main / math.sqrt(DH)
        i_main = igate_preact[:, :, :S_main].view(B, NH, NS_full, CS)
        f_main = fgate_preact[:, :, :S_main].view(B, NH, NS_full, CS)

    # Initialize boundary states
    C = None
    n = None
    m = None
    outputs = []

    if NS_full > 0:
        # Forget gates over time within each chunk (main part)
        log_fgates = torch.nn.functional.logsigmoid(f_main)
        log_fgates_acc = log_fgates.cumsum(dim=3)

        loggates = (i_main - log_fgates_acc)[..., None]
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
        D_max, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)

        stab = torch.maximum(D_max, m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])
        inter_C = (q_main * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()) @ C[:, :, :-1]
        inter_n = (q_main * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()) @ n[:, :, :-1, :, None]

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
    if not TRITON_AVAILABLE:
        # Fallback to simple implementation
        return mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            initial_C=initial_C,
            initial_n=initial_n,
            initial_m=initial_m,
            chunk_size=chunk_size,
            return_last_state=return_last_state,
            eps=eps,
            **kwargs,
        )

    # Lazy import to avoid loading unnecessary modules
    try:
        from .mlstm_triton.torch import mlstm_chunkwise__xl_chunk
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
            chunk_size=chunk_size,
            return_last_state=return_last_state,
            eps=eps,
            **kwargs,
        )

    # Use Triton kernel
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # Initialize states if not provided
    c_initial = initial_C if initial_C is not None else torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_initial = initial_n if initial_n is not None else torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    if n_initial.dim() == 4:
        n_initial = n_initial.squeeze(-1)
    m_initial = initial_m if initial_m is not None else torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    if m_initial.dim() == 4:
        m_initial = m_initial.squeeze(-1)

    # Call the kernel directly
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
    )

    if return_last_state:
        h_seq, (c_last, n_last, m_last) = result
        # Ensure n_last has shape (B, NH, DH, 1) for consistency
        if n_last.dim() == 3:
            n_last = n_last.unsqueeze(-1)
        # Ensure m_last has shape (B, NH, 1, 1) for consistency
        if m_last.dim() == 3:
            m_last = m_last.unsqueeze(-1)
        return h_seq, (c_last, n_last, m_last)
    else:
        return result


__all__ = [
    "TRITON_AVAILABLE",
    "mlstm_chunkwise_simple",
    "mlstm_chunkwise_triton",
    "mlstm_parallel_stabilized_simple",
    "mlstm_recurrent_step_stabilized_simple",
]
