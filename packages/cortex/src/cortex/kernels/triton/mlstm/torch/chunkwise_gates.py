#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""In this file we compute the chunkwise or cumulative gates (i.e. vecA and vecB)
for the forward and backward pass of the mLSTM.
We use the stable formulations, i.e. we avoid subtraction of forget gates.
"""

import torch
from einops import rearrange
from torch.nn.functional import logsigmoid


@torch.compile
def compute_chunkwise_log_gates_vecB_vecA(
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    chunk_size: int,
):
    B, NH, S = vecI.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    _device = vecI.device
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig = logsigmoid(vecF.to(dtype=torch.float32))
    vecF_logsig_chunked = rearrange(vecF_logsig, "b nh (nc l) -> b nh nc l", nc=NC, l=L)
    vecB = vecF_logsig_chunked.cumsum(dim=-1)

    # compute vecA
    vecI_chunked = rearrange(vecI, "b nh (nc l) -> b nh nc l", nc=NC, l=L)
    # unstable vecA computation:
    # vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)
    # stable vecA computation:
    vecA = (
        torch.cat(
            [
                vecF_logsig_chunked[..., 1:].flip(-1).cumsum(-1).flip(-1),
                torch.zeros((B, NH, NC, 1), device=_device, dtype=torch.float32),
            ],
            dim=-1,
        )
        + vecI_chunked
    )  # (B, NH, NC, L)
    return vecB, vecA


@torch.compile
def compute_chunkwise_log_gates_vecB(
    vecF: torch.Tensor,  # (B, NH, S)
    chunk_size: int,
):
    B, NH, S = vecF.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig = logsigmoid(vecF.to(dtype=torch.float32))
    vecF_logsig_chunked = rearrange(vecF_logsig, "b nh (nc l) -> b nh nc l", nc=NC, l=L)
    vecB = vecF_logsig_chunked.cumsum(dim=-1)

    return vecB


# Note: we separate this into a extra function for torch.compile.
# torch.compile will compile this into a single kernel with ca. 0.2 ms runtime (compared to 2.5 ms non-fused kernels)
# for a 1.3B sized model with ctx8192.
@torch.compile
def compute_gate_grads_vecDeltaI_vecDeltaF(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matDeltaQ: torch.Tensor,
    matDeltaK: torch.Tensor,
    vecF: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    #! postprocessing: compute deltaF and deltaI gradients
    ## ? postprocessing
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    matQ = matQ.to(torch.float32)
    matK = matK.to(torch.float32)
    matDeltaQ = matDeltaQ.to(torch.float32)
    matDeltaK = matDeltaK.to(torch.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).to(torch.float32).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing
    # compute deltaI
    # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(-1)
    return vecDeltaI, vecDeltaF
