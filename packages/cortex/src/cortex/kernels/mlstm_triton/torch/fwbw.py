#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ..utils import contiguous, int_or_none, tensor_or_none
from .bw import mlstm_chunkwise_bw
from .fw import mlstm_chunkwise_fw


## PyTorch Autograd Function - Boilerplate
def _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=torch.bfloat16) -> Callable:
    class _mlstm_chunkwise_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,  # (B, NH, S, DHQK)
            matK: torch.Tensor,  # (B, NH, S, DHQK)
            matV: torch.Tensor,  # (B, NH, S, DHV)
            vecI: torch.Tensor,  # (B, NH, S)
            vecF: torch.Tensor,  # (B, NH, S)
            matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
            vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
            scaM_initial: torch.Tensor = None,  # (B, NH, 1)
            qk_scale: float = None,
            return_last_states: bool = False,
            eps: float = 0.0,
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
            recompute_states_in_bw: bool = True,
            reset_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            B, NH, S, DHQK = matQ.shape
            if qk_scale is None:
                qk_scale = DHQK**-0.5

            matH_out, vecN_out, vecM_out, last_states, all_states = mlstm_chunkwise_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                reset_mask=reset_mask,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                scaM_initial=scaM_initial,
                qk_scale=qk_scale,
                return_last_states=return_last_states,
                return_all_states=(not recompute_states_in_bw),
                chunk_size=chunk_size,
                chunk_size_inter=chunk_size_inter,
                chunk_size_intra=chunk_size_intra,
                siz_b_L_parallel=siz_b_L_parallel,
                siz_b_L_loop=siz_b_L_loop,
                siz_b_DH_parallel=siz_b_DH_parallel,
                siz_b_DH_loop=siz_b_DH_loop,
                num_warps_intra=num_warps_intra,
                num_warps_inter=num_warps_inter,
                num_stages_intra=num_stages_intra,
                num_stages_inter=num_stages_inter,
                output_dtype=matQ.dtype,
                eps=eps,
            )

            if return_last_states:
                (matC_last, vecN_last, scaM_last) = last_states
            else:
                (matC_last, vecN_last, scaM_last) = (None, None, None)

            if all_states is not None:
                matC_all, vecN_all, scaM_all = all_states
            else:
                matC_all, vecN_all, scaM_all = (None, None, None)

            ctx.save_for_backward(
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                scaM_initial,
                matC_all,
                vecN_all,
                scaM_all,
                vecN_out,
                vecM_out,
                torch.tensor(qk_scale),
                torch.tensor(chunk_size),
                tensor_or_none(chunk_size_inter),
                tensor_or_none(chunk_size_intra),
                tensor_or_none(siz_b_L_parallel),
                tensor_or_none(siz_b_L_loop),
                tensor_or_none(siz_b_DH_parallel),
                tensor_or_none(siz_b_DH_loop),
                tensor_or_none(num_warps_intra),
                tensor_or_none(num_warps_inter),
                tensor_or_none(num_stages_intra),
                tensor_or_none(num_stages_inter),
                torch.tensor(eps),
                tensor_or_none(reset_mask),
            )
            return matH_out, matC_last, vecN_last, scaM_last

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, matDeltaH_out, matDeltaC_last, vecDeltaN_last, scaDeltaM_last):
            (
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                scaM_initial,
                matC_all,
                vecN_all,
                scaM_all,
                vecN_out,
                vecM_out,
                qk_scale,
                chunk_size,
                chunk_size_inter,
                chunk_size_intra,
                siz_b_L_parallel,
                siz_b_L_loop,
                siz_b_DH_parallel,
                siz_b_DH_loop,
                num_warps_intra,
                num_warps_inter,
                num_stages_intra,
                num_stages_inter,
                eps,
                reset_mask,
            ) = ctx.saved_tensors

            (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                scaDeltaM_initial,
            ) = mlstm_chunkwise_bw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                scaM_initial=scaM_initial,
                matCstate_all=matC_all,
                vecNstate_all=vecN_all,
                scaMstate_all=scaM_all,
                vecN_out=vecN_out,
                vecM_out=vecM_out,
                matDeltaH_out=matDeltaH_out,
                matDeltaC_last=matDeltaC_last,
                qk_scale=float(qk_scale),
                chunk_size=int(chunk_size),
                chunk_size_inter=int_or_none(chunk_size_inter),
                chunk_size_intra=int_or_none(chunk_size_intra),
                siz_b_L_parallel=int_or_none(siz_b_L_parallel),
                siz_b_L_loop=int_or_none(siz_b_L_loop),
                siz_b_DH_parallel=int_or_none(siz_b_DH_parallel),
                siz_b_DH_loop=int_or_none(siz_b_DH_loop),
                num_warps_intra=int_or_none(num_warps_intra),
                num_warps_inter=int_or_none(num_warps_inter),
                num_stages_intra=int_or_none(num_stages_intra),
                num_stages_inter=int_or_none(num_stages_inter),
                eps=float(eps),
                reset_mask=reset_mask if isinstance(reset_mask, torch.Tensor) else None,
            )

            return (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                scaDeltaM_initial,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # reset_mask has no gradient
            )

    return _mlstm_chunkwise_fwbw


_mlstm_chunkwise_fwbw_float32 = _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=torch.float32)
_mlstm_chunkwise_fwbw_float16 = _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=torch.float16)
_mlstm_chunkwise_fwbw_bfloat16 = _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=torch.bfloat16)


def _get_chunkwise_fwbw_kernel(autocast_kernel_dtype: torch.dtype) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_chunkwise_fwbw_float32
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_chunkwise_fwbw_float16
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_chunkwise_fwbw_bfloat16
    else:
        raise ValueError(f"Unsupported kernel dtype {autocast_kernel_dtype}.")


def mlstm_chunkwise__xl_chunk(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None,  # (B, NH, DHQK)
    m_initial: torch.Tensor = None,  # (B, NH, 1)
    return_last_states: bool = False,
    eps: float = 1e-6,
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
    recompute_states_in_bw: bool = True,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    _mlstm_chunkwise_fwbw = _get_chunkwise_fwbw_kernel(autocast_kernel_dtype)
    matH_out, matC_last, vecN_last, scaM_last = _mlstm_chunkwise_fwbw.apply(
        q,
        k,
        v,
        i,
        f,
        c_initial,
        n_initial,
        m_initial,
        None,  # qk_scale always the default value
        return_last_states,
        eps,
        chunk_size,
        chunk_size_inter,
        chunk_size_intra,
        siz_b_L_parallel,
        siz_b_L_loop,
        siz_b_DH_parallel,
        siz_b_DH_loop,
        num_warps_intra,
        num_warps_inter,
        num_stages_intra,
        num_stages_inter,
        recompute_states_in_bw,
        reset_mask,
    )
    if return_last_states:
        return matH_out, (matC_last, vecN_last, scaM_last)
    else:
        return matH_out
