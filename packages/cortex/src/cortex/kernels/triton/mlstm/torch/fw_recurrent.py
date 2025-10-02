#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch

import triton

from ..triton import mlstm_chunkwise__recurrent_fw_C_kernel
from ..triton.kernel_param_heuristics import get_head_dim_block_size
from ..utils import torch2triton_dtype
from ..utils.kernels import is_power_of_2


def mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    vecI: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    vecLastSegMask: torch.Tensor | None = None,  # (B, NH, NC, L) with 1.0 for last-segment positions
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH, 1)
    chunk_size: int = 64,
    num_stages: int | None = None,
    num_warps: int | None = None,
    save_states_every_nth_chunk: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    L = chunk_size
    assert S % L == 0, "Sequence length must be divisible by chunk size."
    NC = S // L

    assert save_states_every_nth_chunk > 0, "save_states_every_nth_chunk must be positive."
    assert save_states_every_nth_chunk <= NC, "save_states_every_nth_chunk must be <= NC."

    assert is_power_of_2(save_states_every_nth_chunk), (
        f"save_states_every_nth_chunk must be a power of 2. Got {save_states_every_nth_chunk}."
    )

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = get_head_dim_block_size(head_dim=DHQK, min_block_size=64)
    siz_b_DHHV = get_head_dim_block_size(head_dim=DHHV, min_block_size=64)

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK == 64 else 2

    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        assert vecN_initial is not None and scaMinter_initial is not None
        str_matCinitial_B_NH = matC_initial.stride(1)
        str_matCinitial_DHQK = matC_initial.stride(2)
        str_matCinitial_DHHV = matC_initial.stride(3)
        str_vecNinitial_B_NH = vecN_initial.stride(1)
        str_vecNinitial_DHQK = vecN_initial.stride(2)
        str_scaMinterinitial_B_NH = scaMinter_initial.stride(1)
    else:
        str_matCinitial_B_NH = 0
        str_matCinitial_DHQK = 0
        str_matCinitial_DHHV = 0
        str_vecNinitial_B_NH = 0
        str_vecNinitial_DHQK = 0
        str_scaMinterinitial_B_NH = 0

    num_chunks_saved = NC // save_states_every_nth_chunk

    matC_states = torch.empty(
        B,
        NH,
        (num_chunks_saved + 1) * DHQK,
        DHHV,
        device=matK.device,
        dtype=torch.float32,
    )
    vecN_states = torch.empty(
        B,
        NH,
        (num_chunks_saved + 1) * DHQK,
        device=matK.device,
        dtype=torch.float32,
    )
    scaMinter_states = torch.empty(B, NH, (num_chunks_saved + 1), device=matK.device, dtype=torch.float32)

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    # Provide default all-ones last-segment mask when none is given
    if vecLastSegMask is None:
        vecLastSegMask = torch.ones((B, NH, NC, L), device=matK.device, dtype=torch.float32)
    mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK=matK,
        matV=matV,
        vecF=vecF,
        vecI=vecI,
        vecLastSegMask=vecLastSegMask,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaMinter_initial,
        str_matK_B_NH=matK.stride(1),
        str_matK_S=matK.stride(2),
        str_matK_DHQK=matK.stride(3),
        str_matV_B_NH=matV.stride(1),
        str_matV_S=matV.stride(2),
        str_matV_DHHV=matV.stride(3),
        str_vecFI_B_NH=vecF.stride(1),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_scaMinterstates_NC=scaMinter_states.stride(2),
        str_vecLastSegMask_B_NH=vecLastSegMask.stride(1),
        str_vecLastSegMask_NC=vecLastSegMask.stride(2),
        str_matCinitial_B_NH=str_matCinitial_B_NH,
        str_matCinitial_DHQK=str_matCinitial_DHQK,
        str_matCinitial_DHHV=str_matCinitial_DHHV,
        str_vecNinitial_B_NH=str_vecNinitial_B_NH,
        str_vecNinitial_DHQK=str_vecNinitial_DHQK,
        str_scaMinterinitial_B_NH=str_scaMinterinitial_B_NH,
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        DTYPE=torch2triton_dtype(matK.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matC_states, vecN_states, scaMinter_states
