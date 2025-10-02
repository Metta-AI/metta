from __future__ import annotations

from cortex.config import (
    CortexStackConfig,
    mLSTMCellConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    sLSTMCellConfig,
)
from cortex.stacks.base import CortexStack


def build_xlstm_stack(
    d_hidden: int,
    num_blocks: int = 7,
    mlstm_num_heads: int = 4,
    slstm_num_heads: int = 4,
    mlstm_proj_factor: float = 2.0,
    slstm_proj_factor: float = 1.5,
    mlstm_chunk_size: int = 256,
    conv1d_kernel_size: int = 4,
    dropout: float = 0.0,
    post_norm: bool = True,
    block_pattern: str | None = None,
) -> CortexStack:
    """Build an xLSTM stack with alternating mLSTM (PreUp) and sLSTM (PostUp) blocks.

    This creates a stack similar to the base xLSTM architecture where:
    - mLSTM blocks use PreUpBlock (projects up before the cell)
    - sLSTM blocks use PostUpBlock (cell first, then FFN sublayer)

    Args:
        d_hidden: External hidden dimension of the stack
        num_blocks: Total number of blocks in the stack
        mlstm_num_heads: Number of heads for mLSTM cells
        slstm_num_heads: Number of heads for sLSTM cells
        mlstm_proj_factor: Projection factor for mLSTM PreUpBlocks
        slstm_proj_factor: Projection factor for sLSTM PostUpBlocks
        mlstm_chunk_size: Chunk size for mLSTM parallel processing
        conv1d_kernel_size: Kernel size for causal conv preprocessing
        dropout: Dropout rate for sLSTM cells
        post_norm: Whether to apply LayerNorm after all blocks
        block_pattern: Pattern string where '0' = mLSTM, '1' = sLSTM (e.g., "0101010").
                      If None, alternates starting with mLSTM.
    """
    blocks = []

    # Determine block pattern
    if block_pattern is None:
        # Default: alternate starting with mLSTM (0)
        pattern = "".join(["0" if i % 2 == 0 else "1" for i in range(num_blocks)])
    else:
        assert len(block_pattern) == num_blocks, f"Pattern length {len(block_pattern)} != num_blocks {num_blocks}"
        pattern = block_pattern

    for i in range(num_blocks):
        if pattern[i] == "0":
            # mLSTM with PreUpBlock
            # hidden_size is inferred by the stack builder for PreUp blocks as:
            # hidden_size = int(proj_factor * d_hidden)
            cell_config = mLSTMCellConfig(
                hidden_size=None,
                num_heads=mlstm_num_heads,
                chunk_size=mlstm_chunk_size,
                conv1d_kernel_size=conv1d_kernel_size,
            )
            block_config = PreUpBlockConfig(
                cell=cell_config,
                proj_factor=mlstm_proj_factor,
            )
        else:
            # sLSTM with PostUpBlock
            # hidden_size is inferred by the stack builder for PostUp blocks as:
            # hidden_size = d_hidden
            cell_config = sLSTMCellConfig(
                hidden_size=None,
                num_heads=slstm_num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
                dropout=dropout,
            )
            block_config = PostUpBlockConfig(
                cell=cell_config,
                proj_factor=slstm_proj_factor,
            )

        blocks.append(block_config)

    # Build the stack configuration
    stack_config = CortexStackConfig(
        blocks=blocks,
        d_hidden=d_hidden,
        post_norm=post_norm,
    )

    return CortexStack(stack_config)


__all__ = ["build_xlstm_stack"]
