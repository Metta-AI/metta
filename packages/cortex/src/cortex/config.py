from __future__ import annotations

from pydantic import BaseModel, Field


class CellConfig(BaseModel):
    """Base configuration for a memory cell.

    All cell configurations should inherit from this class.
    """

    hidden_size: int = Field(ge=1)

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class LSTMCellConfig(CellConfig):
    """Config for a stateless LSTM cell wrapper.

    Notes
    - `hidden_size` must equal the external stack `d_hidden` unless used in a preup block
      that projects to an inner dimension and back.
    - Always uses batch_first=True internally.
    """

    hidden_size: int = Field(ge=1)
    num_layers: int = Field(default=1, ge=1)
    bias: bool = Field(default=True)
    dropout: float = Field(default=0.0, ge=0.0)
    proj_size: int = Field(default=0, ge=0)


class CausalConv1dConfig(CellConfig):
    """Config for a Causal 1D Convolution cell.

    This cell performs causal convolution with optional channel mixing.
    """

    kernel_size: int = Field(default=4, ge=0)
    causal_conv_bias: bool = Field(default=True)
    channel_mixing: bool = Field(default=False)

    @property
    def feature_dim(self) -> int:
        """Alias for hidden_size for compatibility."""
        return self.hidden_size


class mLSTMCellConfig(CellConfig):
    """Config for a stateful mLSTM (Matrix LSTM) cell.

    The mLSTM cell uses matrix-valued state with input/forget gates
    and supports both parallel and recurrent computation modes.
    """

    hidden_size: int = Field(ge=1)
    num_heads: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=256, ge=1)
    # Always apply conv-in inside the cell (depthwise causal conv)
    conv1d_kernel_size: int = Field(default=4, ge=1)


class sLSTMCellConfig(CellConfig):
    """Config for a stateful sLSTM (Structured LSTM) cell.

    This cell follows the reference sLSTM layer structure with optional
    causal depthwise Conv1d used to form pre-activations for gates.

    The recurrence uses per-head recurrent matrices and maintains four
    state tensors per batch row: y, c, n, m.
    """

    hidden_size: int = Field(ge=1)
    num_heads: int = Field(default=4, ge=1)  # Must be power of 2 for triton to work.
    # Optional depthwise causal conv to precondition inputs (0 disables)
    conv1d_kernel_size: int = Field(default=4, ge=0)
    # Dropout applied to the cell output prior to normalization
    dropout: float = Field(default=0.0, ge=0.0)


class BlockConfig(BaseModel):
    """Base configuration for a cortex block.

    This is the base class that all block configurations should inherit from.
    Specific block types can add their own parameters.
    """

    cell: CellConfig  # Any cell config type

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Get the hidden size that the cell should use for this block.

        Args:
            d_hidden: The external hidden size of the stack

        Returns:
            The hidden size to use for the cell

        This can be overridden by subclasses to compute different sizes.
        """
        return d_hidden


class PassThroughBlockConfig(BlockConfig):
    """Configuration for a passthrough block (no projections)."""

    pass


class PreUpBlockConfig(BlockConfig):
    """Configuration for a pre-up projection block.

    Projects up before the cell, then down after.
    """

    proj_factor: float = Field(default=2.0, gt=0.0)

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """For pre-up blocks, the cell operates on the inner dimension."""
        return int(self.proj_factor * d_hidden)


class PostUpBlockConfig(BlockConfig):
    """Configuration for a post-up projection block.

    Applies cell first, then projects up and down.
    """

    proj_factor: float = Field(default=1.5, gt=0.0)


class AdapterBlockConfig(BlockConfig):
    """Configuration for an adapter block that wraps another block.

    The adapter adds a trainable residual path that is identity at initialization.
    This allows inserting adapters into pretrained models without changing behavior at t=0.
    """

    base_block: BlockConfig  # The block to wrap
    cell: CellConfig | None = None  # Not used for adapters, delegated to base_block
    bottleneck: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    per_channel_gate: bool = Field(default=False)
    activation: str = Field(default="gelu")

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Delegate to wrapped block."""
        return self.base_block.get_cell_hidden_size(d_hidden)


class CortexStackConfig(BaseModel):
    """Recipe for building a cortex stack composed of blocks."""

    blocks: list[BlockConfig]  # Accept any BlockConfig subclass
    d_hidden: int = Field(ge=1)
    post_norm: bool = Field(default=True)


__all__ = [
    "CellConfig",
    "CausalConv1dConfig",
    "LSTMCellConfig",
    "mLSTMCellConfig",
    "sLSTMCellConfig",
    "BlockConfig",
    "PassThroughBlockConfig",
    "PreUpBlockConfig",
    "PostUpBlockConfig",
    "AdapterBlockConfig",
    "CortexStackConfig",
]
