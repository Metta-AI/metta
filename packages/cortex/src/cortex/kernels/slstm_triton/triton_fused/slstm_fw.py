# Maximilian Beck
import torch
import triton
import triton.language as tl
from einops import rearrange
from triton import OutOfResources

from .triton_utils import is_power_of_2, next_multiple_of, torch2triton_dtype

# Dimensions:
# B: batch size
# T: sequence length
# NGI: number of gates that depend on input
# NGR: number of gates that depend on recurrent state
# NH: number of heads
# DH: hidden dimension
# NS: number of states

# Note: NS dimension
# > in NS dimension the first index is the state that is used for recurrent weights


# Note on the kernel:
# we only pass the dimensions and not the stride to the kernel
# inside we compute the strides from the dimensions

# we assume for simplicity: NGR == NGI

ENABLE_AUTOTUNING = False

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16, 32, 64]
        for s in [1]
        for w in [1, 2, 4, 8]
    ]
else:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16]
        for s in [1]
        for w in [4]
    ]


@triton.jit
def triton_tanh(x):
    return (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))


@triton.autotune(configs, key=["siz_B", "T", "B", "NH", "DH"])
@triton.jit
def _forward_sequence_kernel(
    states_initial,  # (NH, NS, B, DH) (order: h c n m)
    Wx,  # (NH, T, DGI, B, DH) (order: i f z o)
    R,  # (NH, NGR, DHin, DHout)
    b,  # (NH, NGI, DH)
    states_all,  # (NH, T, NS, B, DH)
    gates_all,  # (NH, T, NGI, B, DH)
    # dimensions,
    T: tl.constexpr,  # sequence length
    NS: tl.constexpr,  # number of states
    B: tl.constexpr,  # batch size
    NH: tl.constexpr,  # number of heads
    DH: tl.constexpr,  # head dimension
    NGI: tl.constexpr,  # number of gates that depend on input
    NGR: tl.constexpr,  # number of gates that depend on recurrent state
    siz_B: tl.constexpr,  # the number of batches per threadblock
    TK: tl.constexpr,  # tile size along K dimension
    TN: tl.constexpr,  # output-column tile size (must be const)
    OUTPUT_GATES: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    idx_b_NH, idx_b_B = tl.program_id(0), tl.program_id(1)

    ## compute the strides
    str_matWx_NH = T * NGI * B * DH
    str_matWx_T = NGI * B * DH
    str_matStatesAll_NH = (T + 1) * NS * B * DH
    str_matStatesAll_T = NS * B * DH
    str_matGatesAll_NH = T * NGI * B * DH
    str_matGatesAll_T = NGI * B * DH
    ##

    ## load the initial states
    # load initial h state to be used for the recurrence
    matHtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matHtrans = tl.load(matHtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial c state
    matCtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matCtrans = tl.load(matCtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial n state
    matNtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matNtrans = tl.load(matNtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial m state
    matMtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 3 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matMtrans = tl.load(matMtrans_initial_ptr).to(tl.float32)  # (B, DH)

    ## store initial states
    # store initial h state in states all
    matHtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matHtrans_initial_store_ptr, matHtrans.to(DTYPE))
    # store initial c state in states all
    matCtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matCtrans_initial_store_ptr, matCtrans.to(DTYPE))
    # store initial n state in states all
    matNtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matNtrans_initial_store_ptr, matNtrans.to(DTYPE))
    # store initial m state in states all
    matMtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 3 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matMtrans_initial_store_ptr, matMtrans.to(DTYPE))

    ## recurrent weights will be processed in output-column tiles to reduce shared memory usage

    ## bias base pointers (tile-wise loads later)
    b_i_base = b + idx_b_NH * NGI * DH + 0 * DH
    b_f_base = b + idx_b_NH * NGI * DH + 1 * DH
    b_z_base = b + idx_b_NH * NGI * DH + 2 * DH
    b_o_base = b + idx_b_NH * NGI * DH + 3 * DH

    for idx_t in range(T):
        # Tile across output columns N with constant TN

        for n0 in range(0, DH, TN):
            # Load feed-forward tiles
            matIx_tile_ptr = tl.make_block_ptr(
                base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 0 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matIx_tile = tl.load(matIx_tile_ptr)

            matFx_tile_ptr = tl.make_block_ptr(
                base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 1 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matFx_tile = tl.load(matFx_tile_ptr)

            matZx_tile_ptr = tl.make_block_ptr(
                base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 2 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matZx_tile = tl.load(matZx_tile_ptr)

            matOx_tile_ptr = tl.make_block_ptr(
                base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 3 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matOx_tile = tl.load(matOx_tile_ptr)

            # Load R tiles (DH x TN) for each gate
            matR_i_tile_ptr = tl.make_block_ptr(
                base=R + idx_b_NH * DH * NGR * DH + 0 * DH * DH,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(0, n0),
                block_shape=(DH, TN),
                order=(0, 1),
            )
            matR_i_tile = tl.load(matR_i_tile_ptr)

            matR_f_tile_ptr = tl.make_block_ptr(
                base=R + idx_b_NH * DH * NGR * DH + 1 * DH * DH,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(0, n0),
                block_shape=(DH, TN),
                order=(0, 1),
            )
            matR_f_tile = tl.load(matR_f_tile_ptr)

            matR_z_tile_ptr = tl.make_block_ptr(
                base=R + idx_b_NH * DH * NGR * DH + 2 * DH * DH,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(0, n0),
                block_shape=(DH, TN),
                order=(0, 1),
            )
            matR_z_tile = tl.load(matR_z_tile_ptr)

            matR_o_tile_ptr = tl.make_block_ptr(
                base=R + idx_b_NH * DH * NGR * DH + 3 * DH * DH,
                shape=(DH, DH),
                strides=(DH, 1),
                offsets=(0, n0),
                block_shape=(DH, TN),
                order=(0, 1),
            )
            matR_o_tile = tl.load(matR_o_tile_ptr)

            # Compute recurrent contributions for tile: (B, TN)
            matRh_i_tile = tl.dot(matHtrans.to(DTYPE), matR_i_tile)
            matRh_f_tile = tl.dot(matHtrans.to(DTYPE), matR_f_tile)
            matRh_z_tile = tl.dot(matHtrans.to(DTYPE), matR_z_tile)
            matRh_o_tile = tl.dot(matHtrans.to(DTYPE), matR_o_tile)

            # Bias tiles
            cols = n0 + tl.arange(0, TN)
            vecBi_tile = tl.load(b_i_base + cols)
            vecBf_tile = tl.load(b_f_base + cols)
            vecBz_tile = tl.load(b_z_base + cols)
            vecBo_tile = tl.load(b_o_base + cols)

            # Current state tiles from time idx_t
            c_t_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t) * str_matStatesAll_T
                + 1 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matC_t_tile = tl.load(c_t_tile_ptr)

            n_t_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t) * str_matStatesAll_T
                + 2 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matN_t_tile = tl.load(n_t_tile_ptr)

            m_t_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t) * str_matStatesAll_T
                + 3 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            matM_t_tile = tl.load(m_t_tile_ptr)

            # Gate preactivations for tile
            matIbar_tile = matIx_tile + matRh_i_tile + vecBi_tile[None, :]
            matFbar_tile = matFx_tile + matRh_f_tile + vecBf_tile[None, :]
            matZbar_tile = matZx_tile + matRh_z_tile + vecBz_tile[None, :]
            matObar_tile = matOx_tile + matRh_o_tile + vecBo_tile[None, :]

            # Pointwise ops per tile
            matLogFplusM_tile = matM_t_tile + tl.log(tl.sigmoid(matFbar_tile))
            # First timestep uses Ibar, thereafter max(Ibar, m + log(sigmoid(Fbar)))
            if idx_t == 0:
                matM_next_tile = matIbar_tile
            else:
                matM_next_tile = tl.maximum(matIbar_tile, matLogFplusM_tile)

            matI_tile = tl.minimum(tl.exp(matIbar_tile - matM_next_tile), 1.0)
            matF_tile = tl.minimum(tl.exp(matLogFplusM_tile - matM_next_tile), 1.0)
            matZ_tile = triton_tanh(matZbar_tile)
            matO_tile = tl.sigmoid(matObar_tile)

            matC_next_tile = matF_tile * matC_t_tile + matI_tile * matZ_tile
            matN_next_tile = matF_tile * matN_t_tile + matI_tile
            matH_next_tile = matO_tile * (matC_next_tile / matN_next_tile)

            # Store next states tiles at time idx_t+1
            h_next_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t + 1) * str_matStatesAll_T
                + 0 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            tl.store(h_next_tile_ptr, matH_next_tile.to(DTYPE))

            c_next_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t + 1) * str_matStatesAll_T
                + 1 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            tl.store(c_next_tile_ptr, matC_next_tile.to(DTYPE))

            n_next_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t + 1) * str_matStatesAll_T
                + 2 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            tl.store(n_next_tile_ptr, matN_next_tile.to(DTYPE))

            m_next_tile_ptr = tl.make_block_ptr(
                base=states_all
                + idx_b_NH * str_matStatesAll_NH
                + (idx_t + 1) * str_matStatesAll_T
                + 3 * B * DH,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, n0),
                block_shape=(siz_B, TN),
                order=(0, 1),
            )
            tl.store(m_next_tile_ptr, matM_next_tile.to(DTYPE))

            # [optional] store gates per tile
            if OUTPUT_GATES:
                gI_tile_ptr = tl.make_block_ptr(
                    base=gates_all
                    + idx_b_NH * str_matGatesAll_NH
                    + idx_t * str_matGatesAll_T
                    + 0 * B * DH,
                    shape=(B, DH),
                    strides=(DH, 1),
                    offsets=(idx_b_B * siz_B, n0),
                    block_shape=(siz_B, TN),
                    order=(0, 1),
                )
                tl.store(gI_tile_ptr, matIbar_tile.to(DTYPE))

                gF_tile_ptr = tl.make_block_ptr(
                    base=gates_all
                    + idx_b_NH * str_matGatesAll_NH
                    + idx_t * str_matGatesAll_T
                    + 1 * B * DH,
                    shape=(B, DH),
                    strides=(DH, 1),
                    offsets=(idx_b_B * siz_B, n0),
                    block_shape=(siz_B, TN),
                    order=(0, 1),
                )
                tl.store(gF_tile_ptr, matFbar_tile.to(DTYPE))

                gZ_tile_ptr = tl.make_block_ptr(
                    base=gates_all
                    + idx_b_NH * str_matGatesAll_NH
                    + idx_t * str_matGatesAll_T
                    + 2 * B * DH,
                    shape=(B, DH),
                    strides=(DH, 1),
                    offsets=(idx_b_B * siz_B, n0),
                    block_shape=(siz_B, TN),
                    order=(0, 1),
                )
                tl.store(gZ_tile_ptr, matZ_tile.to(DTYPE))

                gO_tile_ptr = tl.make_block_ptr(
                    base=gates_all
                    + idx_b_NH * str_matGatesAll_NH
                    + idx_t * str_matGatesAll_T
                    + 3 * B * DH,
                    shape=(B, DH),
                    strides=(DH, 1),
                    offsets=(idx_b_B * siz_B, n0),
                    block_shape=(siz_B, TN),
                    order=(0, 1),
                )
                tl.store(gO_tile_ptr, matO_tile.to(DTYPE))

        # Load next-step h,c,n,m fully for next iteration's recurrent mix
        matHtrans_next_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matHtrans = tl.load(matHtrans_next_ptr).to(tl.float32)

        matCtrans_next_ptr_full = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matCtrans = tl.load(matCtrans_next_ptr_full).to(tl.float32)

        matNtrans_next_ptr_full = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matNtrans = tl.load(matNtrans_next_ptr_full).to(tl.float32)

        matMtrans_next_ptr_full = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matMtrans = tl.load(matMtrans_next_ptr_full).to(tl.float32)


def forward_sequence(
    states_initial: torch.Tensor,  # (NS, B, NH, DH) initial states (h state, c state etc. (h state is used for recurrent weights and is always first))
    Wx: torch.Tensor,  # (B, T, NGI, NH, DH) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    b: torch.Tensor,  # (NGI, NGI, DH) biases
    output_gates_and_states_initial: bool = False,
) -> (
    tuple[
        torch.Tensor,  # (T, NS, B, NH, D) all states
        torch.Tensor,  # (NS, B, NH, D) last state
    ]
    | tuple[
        tuple[
            torch.Tensor,  # (T, NS, B, NH, D) all states
            torch.Tensor,  # (NS, B, NH, D) last state
        ],
        torch.Tensor,  # (T, NGI, B, NH, D) gates
    ]
):
    # support the case where states initial has the time dimension explicitly
    T_dim_explicit = False
    if states_initial.ndim == 5:
        assert (
            states_initial.shape[0] == 1
        ), f"states_initial.shape[0] must be 1: got {states_initial.shape}."
        T_dim_explicit = True
        states_initial = states_initial[0]
    NS, B, NH, DH = states_initial.shape
    _, T, NGI, _, _ = Wx.shape
    NGR, _, _, _ = R.shape
    assert R.shape[1:] == (NH, DH, DH)
    assert b.shape == (NGI, NH, DH)
    assert NS == 4, "sLSTM has 4 states: h, c, n, m."
    assert NGR == 4, "sLSTM has 4 gates: i, f, z, o."
    assert NGR == NGI, "NGR must be equal to NGI."

    dtype = Wx.dtype
    device = Wx.device

    assert R.dtype == dtype, f"dtype mismatch: R.dtype: {R.dtype}, Wx.dtype: {dtype}."

    assert is_power_of_2(DH), f"DH must be a power of 2, got {DH}."
    MIN_BATCH_SIZE = 16  # we need at least 16 batches for tl.dot() (16x16 tensor cores)
    ## batch size padding to be a multiple of MIN_BATCH_SIZE
    effective_B = next_multiple_of(B, MIN_BATCH_SIZE)
    if effective_B != B:
        states_initial = torch.cat(
            [
                states_initial,
                torch.zeros([NS, effective_B - B, NH, DH], device=device, dtype=dtype),
            ],
            dim=1,
        )
        Wx = torch.cat(
            [
                Wx,
                torch.zeros(
                    [effective_B - B, T, NGI, NH, DH], device=device, dtype=dtype
                ),
            ],
            dim=0,
        )
    ## end of batch size padding

    states_all = torch.empty(
        [NH, T + 1, NS, effective_B, DH], device=device, dtype=dtype
    )

    if output_gates_and_states_initial:
        gates_all = torch.empty(
            [NH, T, NGI, effective_B, DH], device=device, dtype=dtype
        )
    else:
        gates_all = None

    # reshape the inputs for the kernel (they must be contiguous)
    states_initial_kshaped = rearrange(
        states_initial, "ns b nh dh -> nh ns b dh"
    ).contiguous()

    Wx_kshaped = rearrange(Wx, "b t ngi nh dh -> nh t ngi b dh").contiguous()
    R_kshaped = rearrange(R, "ngr nh dhout dhin -> nh ngr dhin dhout").contiguous()
    b_kshaped = rearrange(b, "ngi nh dh -> nh ngi dh").contiguous()
    # call the kernel

    def grid(args):
        siz_B = args["siz_B"]
        assert siz_B >= MIN_BATCH_SIZE, "siz_B must be at least 16"
        if siz_B > effective_B:
            # we raise this to skip it in the autotuning
            raise OutOfResources(required=siz_B, limit=effective_B, name="siz_B")
        g = (NH, triton.cdiv(B, siz_B))
        return g

    # Choose an output tile size based on head_dim to reduce shared memory pressure
    TN = 16 if DH >= 128 else 32

    _forward_sequence_kernel[grid](
        states_initial=states_initial_kshaped,
        Wx=Wx_kshaped,
        R=R_kshaped,
        b=b_kshaped,
        states_all=states_all,
        gates_all=gates_all,
        T=T,
        NS=NS,
        B=effective_B,
        NH=NH,
        DH=DH,
        NGI=NGI,
        NGR=NGR,
        TK=32,
        TN=TN,
        OUTPUT_GATES=output_gates_and_states_initial,
        DTYPE=torch2triton_dtype(dtype),
    )

    states_out = rearrange(states_all, "nh t ns b dh -> t ns b nh dh", ns=NS, dh=DH)

    if output_gates_and_states_initial:
        gates_out = rearrange(
            gates_all, "nh t ngi b dh -> t ngi b nh dh", ngi=NGI, dh=DH
        )
        if T_dim_explicit:
            return (states_out, states_out[-1:]), gates_out
        return (states_out, states_out[-1]), gates_out
    else:
        ## remove the padding only when not outputting the gates for backward
        states_out = states_out[:, :, :B, :, :]
        ##
        if T_dim_explicit:
            return states_out[1:], states_out[-1:]
        return states_out[1:], states_out[-1]
