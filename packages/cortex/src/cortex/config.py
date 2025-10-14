"""Configuration classes for Cortex cells, blocks, and stacks."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CellConfig(BaseModel):
    """Base configuration for memory cells with optional hidden size inference."""

    hidden_size: int | None = Field(default=None)

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class LSTMCellConfig(CellConfig):
    """Configuration for standard LSTM cell (batch-first, single layer default)."""

    hidden_size: int | None = Field(default=None)
    num_layers: int = Field(default=1, ge=1)
    bias: bool = Field(default=True)
    dropout: float = Field(default=0.0, ge=0.0)
    proj_size: int = Field(default=0, ge=0)


class CausalConv1dConfig(CellConfig):
    """Configuration for causal 1D convolution with optional channel mixing."""

    kernel_size: int = Field(default=4, ge=0)
    causal_conv_bias: bool = Field(default=True)
    channel_mixing: bool = Field(default=False)

    @property
    def feature_dim(self) -> int:
        """Alias for hidden_size for compatibility."""
        return self.hidden_size


class mLSTMCellConfig(CellConfig):
    """Configuration for Matrix LSTM cell with parallel chunk processing."""

    hidden_size: int | None = Field(default=None)
    num_heads: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=64, ge=1)
    # Always apply conv-in inside the cell (depthwise causal conv)
    conv1d_kernel_size: int = Field(default=4, ge=1)


class sLSTMCellConfig(CellConfig):
    """Configuration for Structured LSTM cell with per-head recurrence."""

    hidden_size: int | None = Field(default=None)
    num_heads: int = Field(default=4, ge=1)  # Must be power of 2 for triton to work.
    # Optional depthwise causal conv to precondition inputs (0 disables)
    conv1d_kernel_size: int = Field(default=4, ge=0)
    # Dropout applied to the cell output prior to normalization
    dropout: float = Field(default=0.0, ge=0.0)


class RTUCellConfig(CellConfig):
    """Configuration for the low-rank Recurrent Trace Unit (RTU) cell.

    The RTU operates on an internal dimension ``hidden_size`` (D == H) and exposes
    a low-rank input map with rank ``rank``. The underlying kernel produces a
    2H-dimensional activation which the Cortex cell projects back to H to fit the
    block interfaces.
    """

    hidden_size: int | None = Field(default=None)
    rank: int = Field(default=16, ge=1)
    activation: str = Field(default="identity")  # one of: silu|relu|tanh|identity
    r_max: float = Field(default=1.0)
    r_min: float = Field(default=0.0)
    max_phase: float = Field(default=6.28)


class AxonsConfig(CellConfig):
    """Configuration for the Axons cell (streaming RTU, diagonal input weights).

    This variant assumes D == H (identity input map) and uses per-channel
    diagonal input weights (w1, w2). The kernel returns a 2H activation that
    the cell projects back to H.
    """

    hidden_size: int | None = Field(default=None)
    activation: str = Field(default="identity")  # one of: silu|relu|tanh|identity
    r_max: float = Field(default=1.0)
    r_min: float = Field(default=0.0)
    max_phase: float = Field(default=6.28)
    # Low‑rank output projection settings: maps 2H -> out_dim via rank `out_rank`.
    # out_dim defaults to H (cell.hidden_size) for compatibility with blocks.
    out_dim: int | None = Field(default=None, ge=1)
    # If None, use the maximum possible rank for a (2H -> out_dim) map, i.e. min(2H, out_dim).
    # If set, the exact value is used.
    out_rank: int | None = Field(default=None)
    # Prefer CUDA seq-allin kernel for short sequences (<= threshold)
    cuda_seq_threshold: int = Field(default=1000, ge=1)


class BlockConfig(BaseModel):
    """Base configuration for cortex blocks."""

    cell: CellConfig  # Any cell config type

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Compute cell hidden size from stack's external dimension."""
        return d_hidden


class PassThroughBlockConfig(BlockConfig):
    """Configuration for a passthrough block (no projections)."""

    pass


class PreUpBlockConfig(BlockConfig):
    """Configuration for pre-upsampling blocks (projects before cell)."""

    proj_factor: float = Field(default=2.0, gt=0.0)

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Cell operates on expanded inner dimension."""
        return int(self.proj_factor * d_hidden)


class PostUpBlockConfig(BlockConfig):
    """Configuration for post-processing blocks (cell then FFN)."""

    proj_factor: float = Field(default=1.5, gt=0.0)


class AdapterBlockConfig(BlockConfig):
    """Configuration for adapter blocks with identity-initialized residual paths."""

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
    """Configuration for building a sequential stack of blocks."""

    blocks: list[BlockConfig]  # Accept any BlockConfig subclass
    d_hidden: int = Field(ge=1)
    post_norm: bool = Field(default=True)


__all__ = [
    "CellConfig",
    "CausalConv1dConfig",
    "LSTMCellConfig",
    "mLSTMCellConfig",
    "sLSTMCellConfig",
    "AxonsConfig",
    "BlockConfig",
    "PassThroughBlockConfig",
    "PreUpBlockConfig",
    "PostUpBlockConfig",
    "AdapterBlockConfig",
    "CortexStackConfig",
]
