#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
import logging
from dataclasses import dataclass

from cortex.kernels.triton.mlstm.utils.kernels import is_power_of_2

LOGGER = logging.getLogger(__name__)


@dataclass
class XLChunkParams:
    """Dataclass to store all chunk size related params for the XL chunk kernel."""

    siz_b_L_parallel: int
    """Parallel block size for the intra chunk part.
    Determines the numer of porgrams over which is parallelized."""
    siz_b_L_loop: int
    """Loop block size for the intra chunk part.
    Determines the number of iterations in the loop."""
    chunk_size_inter: int
    """Inter chunk size."""
    chunk_size_intra: int
    """Intra chunk size."""

    @property
    def save_states_every_nth_chunk(self) -> int:
        """Returns the number of chunks after which the states are saved."""
        return self.chunk_size_intra // self.chunk_size_inter


# The default chunk size for the XL chunk kernel.
DEFAULT_CHUNK_SIZE = 128

# The default block size for the XL chunk kernel.
# Divides the chunk size into (DEFAULT_CHUNK_SIZE // DEFAULT_CHUNK_BLOCK_SIZE) blocks.
DEFAULT_CHUNK_BLOCK_SIZE = 64


def select_heuristic_xl_chunk_kernel_params(
    sequence_length: int,
    target_chunk_size: int | None = None,
) -> XLChunkParams:
    """Heuristic for setting the chunk size params for the XL chunk kernel.
    These params are used to determine the grid and block sizes for the kernel launch and calls the kernel.

    This heuristic tries to choose the best chunk size params for the kernel
    based on the sequence length and target chunk size.

    The default chunk size is 128.
    For sequence_lengths smaller than that use the maximum possible chunk size, which is a power of 2
    (i.e. 16, 32, or 64).
    For sequence_lengths larger than 128, it allows to set the chunk size to a target_chunk_size,
    which must be a multiple of 128.
    This allows to trade-off memory usage for states and compute time.

    Args:
        sequence_length: Sequence length of the inputs. Must be 16, 32, 64, or a multiple of 128.
        target_chunk_size: Target chunk size. Defaults to None.

    Returns:
        XLChunkParams: The chunk size params for the kernel.
    """
    # The default values have been found to be the fastest in benchmarks.
    # We try to be as close to them as possible.
    default_chunk_size_inter = DEFAULT_CHUNK_SIZE  # 256 might be fast too
    default_chunk_size_intra = DEFAULT_CHUNK_SIZE
    default_siz_b_L_parallel = DEFAULT_CHUNK_BLOCK_SIZE
    default_siz_b_L_loop = DEFAULT_CHUNK_BLOCK_SIZE

    # This is a hardware constraint of Triton (tensor core size).
    minimum_divisor_for_sequence_length = 16
    assert sequence_length % minimum_divisor_for_sequence_length == 0, "Sequence length must be divisible by 16."

    if sequence_length < DEFAULT_CHUNK_SIZE:
        # choose the maximum possible chunk size, ignore target_chunk_size
        assert is_power_of_2(sequence_length), "Sequence length must be a power of 2."
        chunk_size_intra = sequence_length
        chunk_size_inter = sequence_length
        siz_b_L_parallel = sequence_length
        siz_b_L_loop = sequence_length

        return XLChunkParams(
            siz_b_L_parallel=sequence_length,
            siz_b_L_loop=sequence_length,
            chunk_size_inter=sequence_length,
            chunk_size_intra=sequence_length,
        )

    else:
        if target_chunk_size is None:
            target_chunk_size = default_chunk_size_intra

        if target_chunk_size < DEFAULT_CHUNK_BLOCK_SIZE:
            LOGGER.warning(
                f"Target chunk size {target_chunk_size} is smaller than the default "
                f"block size {DEFAULT_CHUNK_BLOCK_SIZE}. Setting the all block sizes "
                f"to target_chunk_size."
            )
            chunk_size_inter = min(default_chunk_size_inter, target_chunk_size)
            chunk_size_intra = target_chunk_size
            siz_b_L_loop = target_chunk_size
            siz_b_L_parallel = target_chunk_size

        else:
            assert target_chunk_size % DEFAULT_CHUNK_BLOCK_SIZE == 0, (
                f"Target chunk size must be divisible by the default chunk block size {DEFAULT_CHUNK_BLOCK_SIZE}."
            )
            assert sequence_length % target_chunk_size == 0, (
                f"Sequence length must be divisible by the target chunk size {target_chunk_size}."
            )

            chunk_size_inter = min(default_chunk_size_inter, target_chunk_size)
            chunk_size_intra = target_chunk_size
            siz_b_L_loop = default_siz_b_L_loop
            siz_b_L_parallel = default_siz_b_L_parallel

        return XLChunkParams(
            siz_b_L_parallel=siz_b_L_parallel,
            siz_b_L_loop=siz_b_L_loop,
            chunk_size_inter=chunk_size_inter,
            chunk_size_intra=chunk_size_intra,
        )


def get_xl_chunk_kernel_params(
    sequence_length: int,
    target_chunk_size: int | None = None,
    chunk_size_intra: int | None = None,
    siz_b_L_loop: int | None = None,
    siz_b_L_parallel: int | None = None,
    chunk_size_inter: int | None = None,
) -> XLChunkParams:
    """Validates the given kernel parameters or selects kernel params from heuristic.
    Either specify all kernel parameters or None. If None, the heuristic will be used.
    """

    if chunk_size_intra is not None:
        assert siz_b_L_loop is not None and siz_b_L_parallel is not None and chunk_size_inter is not None, (
            "If you specify the chunk size intra, you must also specify the block sizes."
        )

        assert sequence_length % chunk_size_inter == 0, (
            f"Sequence length {sequence_length} is not divisible by inter chunk size {chunk_size_inter}."
        )

        assert sequence_length % chunk_size_intra == 0, (
            f"Sequence length {sequence_length} is not divisible by intra chunk size {chunk_size_intra}."
        )
        assert chunk_size_inter <= chunk_size_intra, (
            f"chunk_size_inter {chunk_size_inter} must be >= chunk_size_intra {chunk_size_intra}"
        )
        assert chunk_size_intra % chunk_size_inter == 0, (
            f"chunk_size_intra {chunk_size_intra} must be divisible by chunk_size_inter {chunk_size_inter}"
        )

        return XLChunkParams(
            siz_b_L_parallel=siz_b_L_parallel,
            siz_b_L_loop=siz_b_L_loop,
            chunk_size_inter=chunk_size_inter,
            chunk_size_intra=chunk_size_intra,
        )

    else:
        assert siz_b_L_loop is None and siz_b_L_parallel is None and chunk_size_inter is None, (
            "If you do not specify the chunk size intra, you must not specify the block sizes or chunk size inter."
        )

        return select_heuristic_xl_chunk_kernel_params(
            sequence_length=sequence_length, target_chunk_size=target_chunk_size
        )
