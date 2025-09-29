#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the kernel that combines the recurrent and parallel part of the forward pass of the mLSTM chunkwise formulation.
It should allow arbitrary large chunk sizes and head dimensions.
"""

import torch

from ..triton.chunkwise_kernel_param_heuristics import (
    get_xl_chunk_kernel_params,
)
from ..utils import contiguous_noctx
from .fw_parallel import mlstm_chunkwise__parallel_fw_Hintra
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C


@contiguous_noctx
def mlstm_chunkwise_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size: int = 128,
    chunk_size_inter: int | None = None,
    chunk_size_intra: int | None = None,
    siz_b_L_parallel: int | None = None,
    siz_b_L_loop: int | None = None,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    output_dtype: torch.dtype = torch.float32,
    eps: float = 0.0,
) -> tuple[
    torch.Tensor,  # matH_out (B, NH, S, DHHV)
    torch.Tensor,  # vecN_out (B, NH, S)
    torch.Tensor,  # vecM_out (B, NH, S)
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # last_states (matC_states (B, NH, DHQK, DHHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    kernel_chunk_params = get_xl_chunk_kernel_params(
        sequence_length=S,
        target_chunk_size=chunk_size,
        siz_b_L_loop=siz_b_L_loop,
        siz_b_L_parallel=siz_b_L_parallel,
        chunk_size_inter=chunk_size_inter,
        chunk_size_intra=chunk_size_intra,
    )

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecF=vecF,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        chunk_size=kernel_chunk_params.chunk_size_inter,
        save_states_every_nth_chunk=kernel_chunk_params.save_states_every_nth_chunk,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_Hintra(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF,
        matC_states=matC_k_states,
        vecN_states=vecN_k_states,
        scaMinter_states=scaMinter_k_states,
        qk_scale=qk_scale,
        chunk_size=kernel_chunk_params.chunk_size_intra,
        siz_b_LQ=kernel_chunk_params.siz_b_L_parallel,
        siz_b_LKV=kernel_chunk_params.siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        eps=eps,
        output_dtype=output_dtype,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        # Note: we need to make the states contiguous here, because the last states are not contiguous
        # if we return a slice of the larger tensor.
        # For generation afterwards we will use these state tensors and update them in place.
        # For this in place operation the tensor needs to be contiguous.
        # In this case the contigous should result in a copy operation.
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :].contiguous(),
                vecN_k_states[:, :, -DHQK:].contiguous(),
                scaMinter_k_states[:, :, -1:].contiguous(),
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
