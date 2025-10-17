"""xLSTM stack builder with alternating mLSTM and sLSTM blocks."""

from __future__ import annotations

from cortex.config import (
    CortexStackConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    mLSTMCellConfig,
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
    """Build xLSTM stack with alternating mLSTM and sLSTM blocks."""
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
