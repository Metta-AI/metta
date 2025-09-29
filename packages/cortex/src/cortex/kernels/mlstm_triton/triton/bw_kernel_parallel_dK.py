#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the parallel part of the backward pass of the mLSTM chunkwise formulation,
i.e. the "intra-chunk" contribution that computes the deltaK gradients.

The work is partitioned such that there is no limit on either the chunk size or the qk or v dimension.
We use tiling in the chunk dimension L. We tile in Bq and Bkv blocks.
"""

import triton
import triton.language as tl


@triton.jit
def mlstm_chunkwise__parallel_bw_dK_kernel(
    ## input tensor pointers
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    vecA,  # (B, NH, NC, L)
    matCstate_all,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecNstate_all,  # (B, NH, (NC+1) * DHQK)
    scaMstate_all,  # (B, NH, (NC+1))
    vecN_out,  # (B, NH, S) # vecN_combine
    vecM_out,  # (B, NH, S) # vecM_combine
    matDeltaH_out,  # (B, NH, S, DHHV)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    ## output tensor pointers
    matDeltaK,  # (B, NH, S, DHQK)
    qk_scale: tl.constexpr,
    ## strides
    str_matQK_B_NH: tl.constexpr,
    str_matQK_S: tl.constexpr,
    str_matQK_DHQK: tl.constexpr,
    str_matHV_B_NH: tl.constexpr,
    str_matHV_S: tl.constexpr,
    str_matHV_DHHV: tl.constexpr,
    str_vecABI_B_NH: tl.constexpr,
    str_vecABI_NC: tl.constexpr,
    str_matCstate_B_NH: tl.constexpr,
    str_matCstate_NCDHQK: tl.constexpr,
    str_matCstate_DHHV: tl.constexpr,
    str_vecNstate_B_NH: tl.constexpr,
    str_scaMstate_B_NH: tl.constexpr,
    str_vecMN_B_NH: tl.constexpr,
    str_vecMN_S: tl.constexpr,
    ## dimensions
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    ## block sizes
    siz_b_LQ: tl.constexpr,
    siz_b_LKV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    ## other arguments
    DTYPE: tl.constexpr = tl.float32,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 0.0,
):
    # our grid has 4 dimensions: (num_b_DHQK, num_b_LKV, NC, B * NH)
    idx_b_DHQK, idx_b_LKV, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC

    # gate pointers for the current thread block
    vecB_ptr = vecB + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC
    vecI_ptr = vecI + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC

    # load vecB_LKV (siz_b_LKV,)
    vecB_LKV_ptr = vecB_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
    vecB_LKV_val = tl.load(vecB_LKV_ptr).to(tl.float32)
    # load vecI_LKV (siz_b_LKV,)
    vecI_LKV_ptr = vecI_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
    vecI_LKV_val = tl.load(vecI_LKV_ptr).to(tl.float32)

    # ? compute vecAbar for inter chunk contribution
    # load scaM_val (1,)
    scaMinter_k_val = tl.load(scaMstate_all + idx_b_BNH * (NC + 1) + (idx_b_NC + 1)).to(
        tl.float32
    )
    # load vecA (siz_b_LKV,)
    vecA_ptr = (
        vecA
        + idx_b_BNH * str_vecABI_B_NH
        + idx_b_NC * str_vecABI_NC
        + idx_b_LKV * siz_b_LKV
        + tl.arange(0, siz_b_LKV)
    )
    vecA_val = tl.load(vecA_ptr).to(tl.float32)
    # compute vecAbar_val (siz_b_LKV,)
    vecAbar_val = tl.exp(vecA_val - scaMinter_k_val)

    # for causal masking
    b_kv_offset_start = idx_b_LKV * siz_b_LKV
    b_kv_offset_end = (idx_b_LKV + 1) * siz_b_LKV
    b_kv_idxes = b_kv_offset_start + tl.arange(0, siz_b_LKV)

    #! intra chunk contribution
    # init matDeltaK accumulator (siz_b_LKV, siz_b_DHQK)
    matDeltaK_acc = tl.zeros([siz_b_LKV, siz_b_DHQK], dtype=tl.float32)
    ##? loop over siz_b_LQ blocks
    # compute only upper triangular part of the matrix
    idx_b_LQ_start = (idx_b_LKV * siz_b_LKV) // siz_b_LQ
    idx_b_LQ_end = tl.cdiv(L, siz_b_LQ)
    for idx_b_LQ in range(idx_b_LQ_start, idx_b_LQ_end, 1):
        ## compute matDeltaSbar block (siz_b_LQ, siz_b_LKV) -> matDeltaSbar^T (siz_b_LKV, siz_b_LQ)
        ## init matDeltaSbar^T block accumulator (siz_b_LKV, siz_b_LQ)
        matDeltaSbar_trans_acc = tl.zeros([siz_b_LKV, siz_b_LQ], dtype=tl.float32)
        ###? loop over siz_b_DHQK blocks
        for idx_b_DHHV in range(tl.cdiv(DHHV, siz_b_DHHV)):
            ### load matV (non-transposed) (siz_b_LKV, siz_b_DHQK)
            matV_ptr = tl.make_block_ptr(
                base=matV + idx_b_BNH * str_matHV_B_NH,
                shape=(S, DHHV),
                strides=(str_matHV_S, str_matHV_DHHV),
                offsets=(
                    idx_b_NC * L + idx_b_LKV * siz_b_LKV,
                    idx_b_DHHV * siz_b_DHHV,
                ),
                block_shape=(siz_b_LKV, siz_b_DHHV),
                order=(1, 0),
            )
            matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)

            #! inter chunk contribution
            # compute this only on the first iteration
            if idx_b_LQ == idx_b_LQ_start:
                # # compute matKbar (siz_b_LKV, siz_b_DHQK)
                # matKbar_val = matK_val * vecAbar_val[:, None]

                # load matDeltaC_trans (transposed) (siz_b_DHHV, siz_b_DHQK)
                # (idx_b_NC + 1) since matDeltaC_states contains all state delta errors also for the initial state (i.e. NC+1)
                # and in this kernel we take only the last NC states (we do not consider the initial state delta error)
                matDeltaC_trans_ptr = tl.make_block_ptr(
                    base=matDeltaC_states
                    + idx_b_BNH * str_matCstate_B_NH
                    + (idx_b_NC + 1) * DHQK * DHHV,
                    shape=(DHHV, DHQK),
                    strides=(str_matCstate_DHHV, str_matCstate_NCDHQK),
                    offsets=(idx_b_DHHV * siz_b_DHHV, idx_b_DHQK * siz_b_DHQK),
                    block_shape=(siz_b_DHHV, siz_b_DHQK),
                    order=(0, 1),
                )
                matDeltaC_trans_val = tl.load(
                    matDeltaC_trans_ptr, boundary_check=(0, 1)
                ).to(DTYPE)

                # compute matDeltaKbar_inter (siz_b_LKV, siz_b_DHHV)
                matDeltaKbar_inter_val = tl.dot(matV_val, matDeltaC_trans_val)

                # compute matDeltaK_inter (siz_b_LKV, siz_b_DHHV)
                matDeltaK_acc += matDeltaKbar_inter_val * vecAbar_val[:, None]

            ### load matDeltaH_trans (transposed) (siz_b_DHHV, siz_b_LQ)
            matDeltaH_trans_ptr = tl.make_block_ptr(
                base=matDeltaH_out + idx_b_BNH * str_matHV_B_NH,
                shape=(DHHV, S),
                strides=(str_matHV_DHHV, str_matHV_S),
                offsets=(idx_b_DHHV * siz_b_DHHV, idx_b_NC * L + idx_b_LQ * siz_b_LQ),
                block_shape=(siz_b_DHHV, siz_b_LQ),
                order=(0, 1),
            )
            matDeltaH_trans_val = tl.load(
                matDeltaH_trans_ptr, boundary_check=(0, 1)
            ).to(tl.float32)

            # load vecN_out (siz_b_LQ,)
            vecN_out_ptr = (
                vecN_out
                + idx_b_BNH * str_vecMN_B_NH
                + idx_b_NC * L
                + idx_b_LQ * siz_b_LQ
                + tl.arange(0, siz_b_LQ)
            )
            vecN_out_val = tl.load(vecN_out_ptr).to(tl.float32)

            # compute matDeltaH_intra_trans (siz_b_DHHV, siz_b_LQ)
            matDeltaH_trans_val = matDeltaH_trans_val / (vecN_out_val[None, :] + EPS)

            ### compute matDeltaSbar^T (siz_b_LKV, siz_b_LQ)
            matDeltaSbar_trans_acc += tl.dot(matV_val, matDeltaH_trans_val.to(DTYPE))

            ###? end siz_b_DHQK loop

        ### compute matD tile (siz_b_LQ, siz_b_LKV) -> matD^T (siz_b_LKV, siz_b_LQ)
        # load vecB_LQ (siz_b_LQ,)
        vecB_LQ_ptr = vecB_ptr + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
        vecB_LQ_val = tl.load(vecB_LQ_ptr).to(tl.float32)

        # construct gate matrix matDtilde (siz_b_LQ, siz_b_LKV)
        matDtilde_val = (
            vecB_LQ_val[:, None] - vecB_LKV_val[None, :] + vecI_LKV_val[None, :]
        )

        b_q_offset = idx_b_LQ * siz_b_LQ
        # causal masking if on the diagonal
        if b_kv_offset_end >= b_q_offset:
            b_q_idxes = b_q_offset + tl.arange(0, siz_b_LQ)
            mask = b_q_idxes[:, None] >= b_kv_idxes[None, :]
            matDtilde_val = tl.where(mask, matDtilde_val, -float("inf"))

        # load vecM_out (siz_b_LQ,)
        vecM_out_ptr = (
            vecM_out
            + idx_b_BNH * str_vecMN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        vecM_out_val = tl.load(vecM_out_ptr).to(tl.float32)

        # compute matD^T (siz_b_LKV, siz_b_LQ)
        matD_trans_val = tl.trans(tl.exp(matDtilde_val - vecM_out_val[:, None]))
        ### end compute matD tile

        # compute matDeltaS^T (siz_b_LKV, siz_b_LQ)
        matDeltaS_trans_val = matDeltaSbar_trans_acc * qk_scale * matD_trans_val

        # load matQ (siz_b_LQ, siz_b_DHQK)
        matQ_ptr = tl.make_block_ptr(
            base=matQ + idx_b_BNH * str_matQK_B_NH,
            shape=(S, DHQK),
            strides=(str_matQK_S, str_matQK_DHQK),
            offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHQK * siz_b_DHQK),
            block_shape=(siz_b_LQ, siz_b_DHQK),
            order=(1, 0),
        )
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)

        ## accumulate matDeltaK (siz_b_LKV, siz_b_DHQK)
        matDeltaK_acc += tl.dot(matDeltaS_trans_val.to(DTYPE), matQ_val.to(DTYPE))
        ##? end siz_b_LQ loop

    # store matDeltaK (siz_b_LQK, siz_b_DHQK)
    matDeltaK_ptr = tl.make_block_ptr(
        base=matDeltaK + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHQK * siz_b_DHQK),
        block_shape=(siz_b_LKV, siz_b_DHQK),
        order=(1, 0),
    )
    tl.store(matDeltaK_ptr, matDeltaK_acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))
