#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os

import torch
import triton

from ..triton import mlstm_chunkwise__parallel_fw_Hintra_kernel
from ..triton.kernel_param_heuristics import get_head_dim_block_size
from ..utils import torch2triton_dtype
from ..utils.kernels import is_power_of_2
from .chunkwise_gates import compute_chunkwise_log_gates_vecB


def mlstm_chunkwise__parallel_fw_Hintra(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    # these are all the states at every chunk, (we only use NC states up to the last chunk, i.e. :-1)
    matC_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, (NC+1))
    vecSegId: torch.Tensor | None = None,  # (B, NH, NC, L) int32 segment ids (inclusive)
    qk_scale: float = None,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,  # DHHV blocksize for each thread block
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 1e-6,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S), vecM_out (B, NH, S)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LQ
    chunk loop size:            siz_b_LKV
    head dim parallel size:     siz_b_DHHV
    head dim loop size:         siz_b_DHQK
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    assert S % chunk_size == 0, f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # Head-dim tiling (loop for Q/K, parallel for V)
    siz_b_DHQK = get_head_dim_block_size(head_dim=DHQK, min_block_size=64) if siz_b_DHQK is None else siz_b_DHQK
    siz_b_DHHV = get_head_dim_block_size(head_dim=DHHV, min_block_size=128) if siz_b_DHHV is None else siz_b_DHHV

    # Soft shared-memory cap to avoid Triton OutOfResources on GPUs with ~100KB SMEM (e.g., T4/A10).
    # The dominant SMEM consumer is the float32 accumulator `matH_intra_acc` of shape (siz_b_LQ, siz_b_DHHV).
    # Keep a safety margin for other temporaries by budgeting only a fraction of the available SMEM.
    smem_soft_limit = int(os.environ.get("CORTEX_TRITON_SMEM_SOFT_LIMIT", str(96 * 1024)))  # bytes
    bytes_per_acc = 4  # accumulators are float32 in-kernel
    # Account for TWO accumulators of shape (siz_b_LQ, siz_b_DHHV): intra + inter
    denom = max(1, 2 * siz_b_DHHV * bytes_per_acc)
    # Compute a safe upper bound for `siz_b_LQ` given current `siz_b_DHHV`
    max_lq_by_smem = max(16, (smem_soft_limit // denom))
    # Round down to a power of two (Triton block shapes must be power of two)
    pow2 = 1
    while (pow2 << 1) <= max_lq_by_smem:
        pow2 <<= 1
    max_lq_by_smem = max(16, pow2)
    # Apply caps
    if siz_b_LQ > max_lq_by_smem:
        siz_b_LQ = max_lq_by_smem
    # Ensure `siz_b_LKV` stays <= `siz_b_LQ` and divides it; fall back to 32 or 16 when needed
    siz_b_LKV = min(siz_b_LKV, siz_b_LQ)
    if siz_b_LQ % siz_b_LKV != 0:
        # choose the largest divisor of siz_b_LQ among {64, 32, 16}
        for cand in (64, 32, 16):
            if siz_b_LQ % cand == 0 and cand <= siz_b_LKV:
                siz_b_LKV = cand
                break
        else:
            siz_b_LKV = 16

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LKV <= siz_b_LQ, "siz_b_LKV must be less than or equal to siz_b_LQ"
    assert siz_b_LQ % siz_b_LKV == 0, "siz_b_LQ must be divisible by siz_b_LKV"
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)
    num_b_LQ = triton.cdiv(L, siz_b_LQ)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK >= 64 else 2

    matH_out = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=matQ.dtype)
    vecN_out = torch.empty(B, NH, S, device=matQ.device, dtype=output_dtype)
    vecM_out = torch.empty(B, NH, S, device=matQ.device, dtype=output_dtype)

    vecB = compute_chunkwise_log_gates_vecB(vecF=vecF, chunk_size=chunk_size)
    if vecSegId is None:
        vecSegId = torch.zeros((B, NH, NC, L), device=matQ.device, dtype=torch.int32)

    grid = (num_b_DHHV, num_b_LQ, NC * B * NH)
    # print("grid(num_b_DHHV, num_b_LQ, NC*B*NH)", grid)
    mlstm_chunkwise__parallel_fw_Hintra_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        vecI=vecI,
        vecB=vecB,
        vecSegId=vecSegId,
        matHout=matH_out,
        vecNout=vecN_out,
        vecMout=vecM_out,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_vecSegId_B_NH=vecSegId.stride(1),
        str_vecSegId_NC=vecSegId.stride(2),
        str_vecSegId_L=vecSegId.stride(3),
        str_vecMN_B_NH=vecN_out.stride(1),
        str_vecMN_S=vecN_out.stride(2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_LQ=siz_b_LQ,
        siz_b_LKV=siz_b_LKV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=torch2triton_dtype(matQ.dtype),
        OUTPUT_DTYPE=torch2triton_dtype(output_dtype),
        MINIMUM_MAX_VAL=-10.0,
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matH_out, vecN_out, vecM_out
