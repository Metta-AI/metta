#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the parallel part of the backward pass of the mLSTM chunkwise formulation,
i.e. the "intra-chunk" contribution that computes the deltaQ gradients.

The work is partitioned such that there is no limit on either the chunk size or the qk or v dimension.
We use tiling in the chunk dimension L. We tile in Bq and Bkv blocks.
"""

import triton
import triton.language as tl


@triton.jit
def mlstm_chunkwise__parallel_bw_dQ_kernel(
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
    matDeltaQ,  # (B, NH, S, DHQK)
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
    # our grid has 4 dimensions: (num_b_DHQK, num_b_LQ, NC, B * NH)
    idx_b_DHQK, idx_b_LQ, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC

    # gate pointers for the current thread block
    vecB_ptr = vecB + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC
    vecI_ptr = vecI + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC

    # load vecN_out (siz_b_LQ,)
    vecN_out_ptr = vecN_out + idx_b_BNH * str_vecMN_B_NH + idx_b_NC * L + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
    vecN_out_val = tl.load(vecN_out_ptr).to(tl.float32)

    # ? compute vecBbar for inter chunk contribution
    # load vecB_LQ (siz_b_LQ,)
    vecB_LQ_ptr = vecB_ptr + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
    vecB_LQ_val = tl.load(vecB_LQ_ptr).to(tl.float32)
    # load scaM_km1_val (1,)
    # k-1 corresponds to idx_b_NC
    scaMinter_km1_val = tl.load(scaMstate_all + idx_b_BNH * (NC + 1) + (idx_b_NC)).to(tl.float32)
    # load vecM_out (siz_b_LQ,)
    vecM_out_ptr = vecM_out + idx_b_BNH * str_vecMN_B_NH + idx_b_NC * L + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
    vecM_out_val = tl.load(vecM_out_ptr).to(tl.float32)
    # compute vecBbar (siz_b_LQ,)
    vecBbar_val = tl.exp(vecB_LQ_val + scaMinter_km1_val - vecM_out_val)
    # ? end compute vecBbar

    # for causal masking
    b_q_offset = idx_b_LQ * siz_b_LQ
    b_q_idxes = b_q_offset + tl.arange(0, siz_b_LQ)

    #! intra chunk contribution
    # init matDeltaQ accumulator (siz_b_LQ, siz_b_DHQK)
    matDeltaQ_acc = tl.zeros([siz_b_LQ, siz_b_DHQK], dtype=tl.float32)
    ##? loop over siz_b_LKV blocks
    # only compute the lower triangular part
    idx_b_LKV_end = ((idx_b_LQ + 1) * siz_b_LQ) // siz_b_LKV
    for idx_b_LKV in range(idx_b_LKV_end):
        ## compute matDeltaSbar tile (siz_b_LQ, siz_b_LKV)
        ## init matDeltaSbar tile accumulator (siz_b_LQ, siz_b_LKV)
        matDeltaSbar_acc = tl.zeros([siz_b_LQ, siz_b_LKV], dtype=tl.float32)
        ###? loop over siz_b_DHQK blocks
        for idx_b_DHHV in range(tl.cdiv(DHHV, siz_b_DHHV)):
            ### load matDeltaH (non-transposed) (siz_b_LQ, siz_b_DHHV)
            matDeltaH_ptr = tl.make_block_ptr(
                base=matDeltaH_out + idx_b_BNH * str_matHV_B_NH,
                shape=(S, DHHV),
                strides=(str_matHV_S, str_matHV_DHHV),
                offsets=(
                    idx_b_NC * L + idx_b_LQ * siz_b_LQ,
                    idx_b_DHHV * siz_b_DHHV,
                ),
                block_shape=(siz_b_LQ, siz_b_DHHV),
                order=(1, 0),
            )
            matDeltaH_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(DTYPE)

            #! inter chunk contribution
            # compute this only on the first iteration
            if idx_b_LKV == 0:
                # load matC_km1_trans (transposed) (siz_b_DHHV, siz_b_DHQK)
                # idx_b_NC corresponds to k-1
                matC_km1_trans_ptr = tl.make_block_ptr(
                    base=matCstate_all + idx_b_BNH * str_matCstate_B_NH + idx_b_NC * DHQK * DHHV,
                    shape=(DHHV, DHQK),
                    strides=(str_matCstate_DHHV, str_matCstate_NCDHQK),
                    offsets=(idx_b_DHHV * siz_b_DHHV, idx_b_DHQK * siz_b_DHQK),
                    block_shape=(siz_b_DHHV, siz_b_DHQK),
                    order=(0, 1),
                )
                matC_trans_val = tl.load(matC_km1_trans_ptr, boundary_check=(0, 1)).to(DTYPE)

                # compute matDeltaQbar_inter (siz_b_LQ, siz_b_DHQK)
                matDeltaQbar_inter_val = tl.dot(matDeltaH_val, matC_trans_val) / (vecN_out_val[:, None] + EPS)

                # compute matDeltaQ_inter (siz_b_LQ, siz_b_DHQK)
                matDeltaQ_acc += matDeltaQbar_inter_val * vecBbar_val[:, None] * qk_scale

            ### load matV_trans (transposed) (siz_b_DHHV, siz_b_LKV)
            matV_trans_ptr = tl.make_block_ptr(
                base=matV + idx_b_BNH * str_matHV_B_NH,
                shape=(DHHV, S),
                strides=(str_matHV_DHHV, str_matHV_S),
                offsets=(
                    idx_b_DHHV * siz_b_DHHV,
                    idx_b_NC * L + idx_b_LKV * siz_b_LKV,
                ),
                block_shape=(siz_b_DHHV, siz_b_LKV),
                order=(0, 1),
            )
            matV_trans_val = tl.load(matV_trans_ptr, boundary_check=(0, 1)).to(DTYPE)

            ### compute matDeltaSbar (siz_b_LQ, siz_b_LKV)
            matDeltaSbar_acc += tl.dot(matDeltaH_val, matV_trans_val)

            ###? end siz_b_DHQK loop

        ###? compute matD tile (siz_b_LQ, siz_b_LKV)
        # load vecI_LKV (siz_b_LKV,)
        vecI_LKV_ptr = vecI_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecI_LKV_val = tl.load(vecI_LKV_ptr).to(tl.float32)

        # load vecB_LQ (siz_b_LQ,)
        vecB_LKV_ptr = vecB_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecB_LKV_val = tl.load(vecB_LKV_ptr).to(tl.float32)

        # construct gate matrix matDtilde (siz_b_LQ, siz_b_LKV)
        matDtilde_val = vecB_LQ_val[:, None] - vecB_LKV_val[None, :] + vecI_LKV_val[None, :]

        b_kv_offset = idx_b_LKV * siz_b_LKV
        # causal masking if on the diagonal
        if b_kv_offset >= b_q_offset:
            b_kv_idxes = b_kv_offset + tl.arange(0, siz_b_LKV)
            mask = b_q_idxes[:, None] >= b_kv_idxes[None, :]
            matDtilde_val = tl.where(mask, matDtilde_val, -float("inf"))

        # compute matD (siz_b_LQ, siz_b_LKV)
        matD_val = tl.exp(matDtilde_val - vecM_out_val[:, None])
        ###? end compute matD tile

        # divide by vecN_out_val (siz_b_LQ,)
        # Note: we change the order of matrix multiply and division here.
        # Actually we would compute matDeltaH / vecN_out_val first and then multiply
        # We do this here to avoid the division in the inner loop, for better performance
        # It should not cause too much numerical deviations
        matDeltaSbar_acc = matDeltaSbar_acc / (vecN_out_val[:, None] + EPS)

        # compute matDeltaS (siz_b_LQ, siz_b_LKV)
        matDeltaS_val = matDeltaSbar_acc * qk_scale * matD_val

        # load matK (siz_b_LKV, siz_b_DHQK)
        matK_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * str_matQK_B_NH,
            shape=(S, DHQK),
            strides=(str_matQK_S, str_matQK_DHQK),
            offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHQK * siz_b_DHQK),
            block_shape=(siz_b_LKV, siz_b_DHQK),
            order=(1, 0),
        )
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)

        ## accumulate matDeltaK (siz_b_LQ, siz_b_DHQK)
        matDeltaQ_acc += tl.dot(matDeltaS_val.to(DTYPE), matK_val)
        ##? end siz_b_LQ loop

    # store matDeltaQ (siz_b_LQK, siz_b_DHQK)
    matDeltaQ_ptr = tl.make_block_ptr(
        base=matDeltaQ + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHQK * siz_b_DHQK),
        block_shape=(siz_b_LQ, siz_b_DHQK),
        order=(1, 0),
    )
    tl.store(matDeltaQ_ptr, matDeltaQ_acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))
