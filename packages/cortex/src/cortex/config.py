"""Configuration classes for Cortex cells, blocks, and stacks with type tags for JSON serialization."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field, SerializeAsAny, field_validator


class CellConfig(BaseModel):
    """Base configuration for memory cells with optional hidden size inference."""

    hidden_size: int | None = Field(default=None)

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class LSTMCellConfig(CellConfig):
    """Configuration for standard LSTM cell (batch-first, single layer default)."""

    cell_type: str = "lstm"
    hidden_size: int | None = Field(default=None)
    num_layers: int = Field(default=1, ge=1)
    bias: bool = Field(default=True)
    dropout: float = Field(default=0.0, ge=0.0)
    proj_size: int = Field(default=0, ge=0)


class CausalConv1dConfig(CellConfig):
    """Configuration for causal 1D convolution with optional channel mixing."""

    cell_type: str = "cconv"
    kernel_size: int = Field(default=4, ge=0)
    causal_conv_bias: bool = Field(default=True)
    channel_mixing: bool = Field(default=False)

    @property
    def feature_dim(self) -> int:
        """Alias for hidden_size for compatibility."""
        return self.hidden_size


class mLSTMCellConfig(CellConfig):
    """Configuration for Matrix LSTM cell with parallel chunk processing."""

    cell_type: str = "mlstm"
    hidden_size: int | None = Field(default=None)
    num_heads: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=64, ge=1)
    # Always apply conv-in inside the cell (depthwise causal conv)
    conv1d_kernel_size: int = Field(default=4, ge=1)
    # Use AxonLayer for gating projections (3H -> NH)
    use_axon_layer: bool = Field(default=False)
    # Optionally also use AxonLayers for Q/K/V (defaults to conv path)
    use_axon_qkv: bool = Field(default=False)
    # Rank for AxonLayer low-rank output map when enabled (None → use Axon default)
    # Optional Axon configs to customize AxonLayer behavior when enabled.
    # If provided, these will be forwarded to the internal AxonLayer instances
    # for the gates (3H->NH) and the optional QKV path (H->H). IO sizes and
    # activation will be overridden by the AxonLayer wrapper as needed.
    axon_layer_config: AxonConfig | None = Field(default=None)
    axon_qkv_config: AxonConfig | None = Field(default=None)
    # When use_axon_qkv is enabled, Q/K share a layer and V has its own layer.


class sLSTMCellConfig(CellConfig):
    """Configuration for Structured LSTM cell with per-head recurrence."""

    cell_type: str = "slstm"
    hidden_size: int | None = Field(default=None)
    num_heads: int = Field(default=4, ge=1)  # Must be power of 2 for triton to work.
    # Optional depthwise causal conv to precondition inputs (0 disables)
    conv1d_kernel_size: int = Field(default=4, ge=0)
    # Dropout applied to the cell output prior to normalization
    dropout: float = Field(default=0.0, ge=0.0)
    # Use AxonLayer-based gates instead of block-diagonal Linear expand
    use_axon_layer: bool = Field(default=False)
    # Optional Axon config forwarded to fused gate AxonLayers when enabled.
    axon_layer_config: AxonConfig | None = Field(default=None)


class XLCellConfig(CellConfig):
    """Configuration for Transformer-XL style attention cell."""

    cell_type: str = "xl"
    hidden_size: int | None = Field(default=None)
    n_heads: int = Field(default=4, ge=1)
    head_dim: int | None = Field(default=None, ge=1)
    mem_len: int = Field(default=128, ge=0)
    attn_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    out_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    use_bias: bool = Field(default=True)
    # Optional AxonLayer-backed Q/K/V projections
    use_axon_qkv: bool = Field(default=False)
    axon_qkv_config: AxonConfig | None = Field(default=None)


class AxonConfig(CellConfig):
    """Configuration for Axon cell with streaming RTU and diagonal input weights."""

    cell_type: str = "axon"
    hidden_size: int | None = Field(default=None)
    activation: str = Field(default="identity")  # one of: silu|relu|tanh|identity
    r_max: float = Field(default=1.0)
    r_min: float = Field(default=0.0)
    max_phase: float = Field(default=6.28)
    # Output projection settings: maps 2H -> out_dim via a single linear layer.
    # out_dim defaults to H (cell.hidden_size) for compatibility with blocks.
    out_dim: int | None = Field(default=None, ge=1)
    # Prefer CUDA seq-allin kernel for short sequences (<= threshold)
    cuda_seq_threshold: int = Field(default=5000, ge=1)
    # Optional SRHT mixer before the kernel (diagonal input map remains in mixed basis)
    use_srht: bool = Field(default=False)
    srht_permute: bool = Field(default=True)
    # Use full‑rank RTU kernel (Wc1/Wc2 in R^{D×H}) instead of diagonal (w1/w2 in R^H).
    # When enabled, AxonCell selects between PyTorch and CUDA full‑rank backends
    # using the same CUDA preference policy. Disabled by default for stability.
    use_fullrank_rtu: bool = Field(default=False)
    # Use an untraced learnable linear input projection (H -> H) instead of SRHT.
    # This projection is applied outside the traced kernel and does not receive
    # cross‑chunk boundary corrections. It is automatically disabled when
    # use_fullrank_rtu=True. It also disables srht if enabled.
    use_untraced_linear: bool = Field(default=True)


class BlockConfig(BaseModel):
    """Base configuration for cortex blocks."""

    # May be overridden to None (e.g., Adapter) in subclasses
    # serialize_as_any=True preserves concrete subclass fields/tags during dumps
    cell: SerializeAsAny[CellConfig | None] = Field(default=None)

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Compute cell hidden size from stack's external dimension."""
        return d_hidden

    # Coerce nested `cell` dicts into the correct CellConfig subclass using tag
    @field_validator("cell", mode="before")
    @classmethod
    def _coerce_cell(cls, value: Any) -> Any:
        if value is None or isinstance(value, CellConfig):
            return value
        if isinstance(value, Mapping):
            tag = value.get("cell_type")
            if not isinstance(tag, str) or not tag:
                return value
            # Import locally to avoid import cycles at module import time
            from cortex.cells.registry import get_cell_config_class

            cfg_cls = get_cell_config_class(tag)
            return cfg_cls.model_validate(value)
        return value


class PassThroughBlockConfig(BlockConfig):
    """Configuration for a passthrough block (no projections)."""

    block_type: str = "passthrough"


class PreUpBlockConfig(BlockConfig):
    """Configuration for pre-upsampling blocks (projects before cell)."""

    block_type: str = "preup"
    proj_factor: float = Field(default=2.0, gt=0.0)
    # When True, applies the activation to the 'a' projection before
    # passing it into the cell (i.e., feeds a_act instead of a). This
    # is automatically disabled for mLSTM cells to avoid changing
    # semantics there.
    activate_cell_input: bool = Field(default=True)

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Cell operates on expanded inner dimension."""
        return int(self.proj_factor * d_hidden)


class PostUpBlockConfig(BlockConfig):
    """Configuration for post-processing blocks (cell then FFN)."""

    block_type: str = "postup"
    proj_factor: float = Field(default=1.5, gt=0.0)


class AdapterBlockConfig(BlockConfig):
    """Configuration for adapter blocks with identity-initialized residual paths."""

    block_type: str = "adapter"
    # Preserve subclass fields/tags on dump
    base_block: SerializeAsAny[BlockConfig]  # The block to wrap
    cell: CellConfig | None = None  # Not used for adapters, delegated to base_block
    bottleneck: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    per_channel_gate: bool = Field(default=False)
    activation: str = Field(default="gelu")

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Delegate to wrapped block."""
        return self.base_block.get_cell_hidden_size(d_hidden)

    # Coerce nested base_block via its tag
    @field_validator("base_block", mode="before")
    @classmethod
    def _coerce_base_block(cls, value: Any) -> Any:
        if isinstance(value, BlockConfig):
            return value
        if isinstance(value, Mapping):
            tag = value.get("block_type")
            if not isinstance(tag, str) or not tag:
                return value
            from cortex.blocks.registry import get_block_config_class

            cfg_cls = get_block_config_class(tag)
            return cfg_cls.model_validate(value)
        return value


class CortexStackConfig(BaseModel):
    """Configuration for a sequential stack of blocks."""

    # Preserve subclass fields/tags for each block on dump
    blocks: list[SerializeAsAny[BlockConfig]]  # Accept any BlockConfig subclass
    d_hidden: int = Field(ge=1)
    post_norm: bool = Field(default=True)
    compile_blocks: bool = Field(default=True)

    # Coerce list items into correct BlockConfig subclass using tag
    @field_validator("blocks", mode="before")
    @classmethod
    def _coerce_blocks(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for item in value:
            if isinstance(item, BlockConfig):
                out.append(item)
                continue
            if isinstance(item, Mapping):
                tag = item.get("block_type")
                if isinstance(tag, str) and tag:
                    from cortex.blocks.registry import get_block_config_class

                    cfg_cls = get_block_config_class(tag)
                    out.append(cfg_cls.model_validate(item))
                    continue
            out.append(item)
        return out


class RouterConfig(BaseModel):
    """Router settings with global prior and optional per-token refinement."""

    # Global prior settings
    d_key: int | None = Field(default=None, ge=1, description="Key/query dim for global prior; defaults to d_hidden.")
    temperature: float = Field(default=1.0, gt=0.0, description="Softmax temperature for the global gate.")
    top_k: int | None = Field(default=None, ge=1, description="If set, keep only top‑k experts in the global prior.")
    use_sqrt_scale: bool = Field(default=True, description="Use 1/sqrt(d_key) (vs 1/d_key) dot‑product scaling.")
    init_scale_wq: float = Field(default=0.0, description="Uniform init scale for Wq; 0 → near‑uniform prior.")
    init_scale_wk: float = Field(default=0.0, description="Uniform init scale for Wk; 0 → near‑uniform prior.")

    # Per-token refinement (optional; disabled when whisper_lambda == 0)
    d_key_local: int | None = Field(default=None, ge=1, description="Key dim for per‑token refiner; defaults to d_key.")
    local_temperature: float = Field(default=1.0, gt=0.0, description="Temperature for token‑refiner logits.")
    whisper_lambda: float = Field(default=0.1, ge=0.0, description="Strength λ of per‑token refinement (0 disables).")
    center_refine: bool = Field(default=True, description="Center token logits over experts to redistribute mass only.")
    restrict_to_topk: bool = Field(default=True, description="Limit refinement to the global top‑k support if set.")

    class Config:
        extra = "allow"


class ColumnBlockConfig(BlockConfig):
    """Column of experts with a shared router."""

    block_type: str = "column"
    experts: list[SerializeAsAny[BlockConfig]]
    router: RouterConfig = Field(default_factory=RouterConfig, description="Router hyperparameters for this column.")
    alpha_col_init: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Initial outer ReZero gain α_col; controls how much gradient from this block flows at init "
            "(small for calm starts)."
        ),
    )

    class Config:
        extra = "allow"

    def get_cell_hidden_size(self, d_hidden: int) -> int:  # type: ignore[override]
        return d_hidden

    @field_validator("experts", mode="before")
    @classmethod
    def _coerce_experts(cls, value):
        if not isinstance(value, list):
            return value
        out: list[BlockConfig] = []
        for item in value:
            if isinstance(item, BlockConfig):
                out.append(item)
                continue
            if isinstance(item, Mapping):
                tag = item.get("block_type")
                if isinstance(tag, str) and tag:
                    from cortex.blocks.registry import get_block_config_class

                    cfg_cls = get_block_config_class(tag)
                    out.append(cfg_cls.model_validate(item))
                    continue
            out.append(item)
        return out


__all__ = [
    "CellConfig",
    "CausalConv1dConfig",
    "LSTMCellConfig",
    "mLSTMCellConfig",
    "sLSTMCellConfig",
    "XLCellConfig",
    "AxonConfig",
    "BlockConfig",
    "PassThroughBlockConfig",
    "PreUpBlockConfig",
    "PostUpBlockConfig",
    "AdapterBlockConfig",
    "CortexStackConfig",
    "RouterConfig",
    "ColumnBlockConfig",
]
