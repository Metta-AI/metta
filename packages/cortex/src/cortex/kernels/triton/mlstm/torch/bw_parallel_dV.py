#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os

import torch
import triton

from cortex.kernels.triton.mlstm.triton import mlstm_chunkwise__parallel_bw_dV_kernel
from cortex.kernels.triton.mlstm.triton.kernel_param_heuristics import get_head_dim_block_size
from cortex.kernels.triton.mlstm.utils import torch2triton_dtype
from cortex.kernels.triton.mlstm.utils.kernels import is_power_of_2


def mlstm_chunkwise__parallel_bw_dV(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecA: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecSegId: torch.Tensor,  # (B, NH, NC, L)
    ## Backward arguments
    matCstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecNstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK)
    scaMstate_all: torch.Tensor,  # (B, NH, (NC+1))
    vecN_out: torch.Tensor,  # (B, NH, S) # vecN_combine
    vecM_out: torch.Tensor,  # (B, NH, S) # vecM_combine
    matDeltaH_out: torch.Tensor,  # (B, NH, S, DHHV)
    matDeltaC_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    ## Other arguments
    qk_scale: float = None,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:  # matDeltaV (B, NH, S, DHHV)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LKV
    chunk loop size:            siz_b_LQ
    head dim parallel size:     siz_b_DHHV
    head dim loop size:         siz_b_DHQK
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    assert S % chunk_size == 0, f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = get_head_dim_block_size(head_dim=DHQK, min_block_size=64) if siz_b_DHQK is None else siz_b_DHQK
    siz_b_DHHV = get_head_dim_block_size(head_dim=DHHV, min_block_size=128) if siz_b_DHHV is None else siz_b_DHHV

    # Shared-memory soft cap: primary accumulator is (siz_b_LKV, siz_b_DHHV) float32.
    # Secondary buffers (LKV x LQ) appear transiently; budget for both by using (DHHV + LQ_small).
    smem_soft_limit = int(os.environ.get("CORTEX_TRITON_SMEM_SOFT_LIMIT", str(96 * 1024)))
    bytes_per = 4

    # Choose a small, power-of-two LQ that divides LKV and reduces temporary sizes
    def pow2_le(x: int) -> int:
        p = 1
        while (p << 1) <= x:
            p <<= 1
        return max(16, p)

    LQ_small_target = pow2_le(min(siz_b_LQ, 32))
    # Account for double-buffered accumulators of shape (LKV, DHHV) plus an (LKV, LQ) temp
    denom = bytes_per * (2 * max(1, DHHV) + LQ_small_target)
    max_lkv = max(16, smem_soft_limit // denom)
    max_lkv = pow2_le(min(max_lkv, L))
    if siz_b_LKV > max_lkv:
        siz_b_LKV = max_lkv
    # Ensure LQ <= LKV and LKV % LQ == 0
    siz_b_LQ = min(siz_b_LQ, siz_b_LKV)
    siz_b_LQ = pow2_le(siz_b_LQ)
    while siz_b_LKV % siz_b_LQ != 0 and siz_b_LQ > 16:
        siz_b_LQ //= 2

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LQ <= siz_b_LKV, "siz_b_LQ must be less than or equal to siz_b_LKV"
    assert siz_b_LKV % siz_b_LQ == 0, "siz_b_LKV must be divisible by siz_b_LQ"

    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)
    num_b_LKV = triton.cdiv(L, siz_b_LKV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if (siz_b_DHQK >= 64 or siz_b_DHHV >= 64) else 2

    matDeltaV = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=output_dtype)
    grid = (num_b_DHHV, num_b_LKV, NC * B * NH)

    mlstm_chunkwise__parallel_bw_dV_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        vecSegId=vecSegId,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        scaMstate_all=scaMstate_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        matDeltaV=matDeltaV,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_vecABI_B_NH=vecB.stride(1),
        str_vecABI_NC=vecB.stride(2),
        str_matCstate_B_NH=matCstate_all.stride(1),
        str_matCstate_NCDHQK=matCstate_all.stride(2),
        str_matCstate_DHHV=matCstate_all.stride(3),
        str_vecNstate_B_NH=vecNstate_all.stride(1),
        str_scaMstate_B_NH=scaMstate_all.stride(1),
        str_vecMN_B_NH=vecN_out.stride(1),
        str_vecMN_S=vecN_out.stride(2),
        str_vecSegId_B_NH=vecSegId.stride(1),
        str_vecSegId_NC=vecSegId.stride(2),
        str_vecSegId_L=vecSegId.stride(3),
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
        EPS=eps,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return matDeltaV
