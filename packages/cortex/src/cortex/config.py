"""Configuration classes for Cortex cells, blocks, and stacks with type tags for JSON serialization."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator


class CellConfig(BaseModel):
    """Base configuration for memory cells with optional hidden size inference."""

    model_config = ConfigDict(extra="allow")

    hidden_size: int | None = Field(default=None)


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
    conv1d_kernel_size: int = Field(default=4, ge=1)
    use_axon_layer: bool = Field(default=False)
    use_axon_qkv: bool = Field(default=False)
    axon_layer_config: AxonConfig | None = Field(default=None)
    axon_qkv_config: AxonConfig | None = Field(default=None)


class sLSTMCellConfig(CellConfig):
    """Configuration for Structured LSTM cell with per-head recurrence."""

    cell_type: str = "slstm"
    hidden_size: int | None = Field(default=None)
    num_heads: int = Field(default=4, ge=1)
    conv1d_kernel_size: int = Field(default=4, ge=0)
    dropout: float = Field(default=0.0, ge=0.0)
    use_axon_layer: bool = Field(default=False)
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
    use_axon_qkv: bool = Field(default=False)
    axon_qkv_config: AxonConfig | None = Field(default=None)


class AGaLiTeCellConfig(CellConfig):
    """Configuration for AGaLiTe attention cell with recurrent discounted state."""

    cell_type: str = "agalite"
    hidden_size: int | None = Field(default=None)
    n_heads: int = Field(default=8, ge=1)
    head_dim: int | None = Field(default=None, ge=1)
    eta: int = Field(default=6, ge=1)
    r: int = Field(default=2, ge=1)
    eps: float = Field(default=1e-5, ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class AxonConfig(CellConfig):
    """Configuration for Axon cell with streaming RTU and diagonal input weights."""

    cell_type: str = "axon"
    hidden_size: int | None = Field(default=None)
    activation: str = Field(default="identity")
    r_max: float = Field(default=1.0)
    r_min: float = Field(default=0.0)
    max_phase: float = Field(default=6.28)
    out_dim: int | None = Field(default=None, ge=1)
    cuda_seq_threshold: int = Field(default=5000, ge=1)
    use_srht: bool = Field(default=False)
    srht_permute: bool = Field(default=True)
    use_fullrank_rtu: bool = Field(default=False)
    use_untraced_linear: bool = Field(default=True)


class BlockConfig(BaseModel):
    """Base configuration for cortex blocks."""

    model_config = ConfigDict(extra="allow")

    cell: SerializeAsAny[CellConfig | None] = Field(default=None)

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Compute cell hidden size from stack's external dimension."""
        return d_hidden

    @field_validator("cell", mode="before")
    @classmethod
    def _coerce_cell(cls, value: Any) -> Any:
        if value is None or isinstance(value, CellConfig):
            return value
        if isinstance(value, Mapping):
            tag = value.get("cell_type")
            if not isinstance(tag, str) or not tag:
                return value
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
    activate_cell_input: bool = Field(default=True)

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Cell operates on expanded inner dimension."""
        return int(self.proj_factor * d_hidden)


class PostUpBlockConfig(BlockConfig):
    """Configuration for post-processing blocks (cell then FFN)."""

    block_type: str = "postup"
    proj_factor: float = Field(default=1.5, gt=0.0)


class PostUpGatedBlockConfig(BlockConfig):
    """Configuration for GRU‑gated post blocks (GTrXL‑style gating)."""

    block_type: str = "postup_gated"
    proj_factor: float = Field(default=1.5, gt=0.0)
    gru_bias: float = Field(default=2.0)


class AdapterBlockConfig(BlockConfig):
    """Configuration for adapter blocks with identity-initialized residual paths."""

    block_type: str = "adapter"
    base_block: SerializeAsAny[BlockConfig]
    cell: CellConfig | None = None
    bottleneck: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    per_channel_gate: bool = Field(default=False)
    activation: str = Field(default="gelu")

    def get_cell_hidden_size(self, d_hidden: int) -> int:
        """Delegate to wrapped block."""
        return self.base_block.get_cell_hidden_size(d_hidden)

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

    blocks: list[SerializeAsAny[BlockConfig]]
    d_hidden: int = Field(ge=1)
    post_norm: bool = Field(default=True)
    compile_blocks: bool = Field(default=True)

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

    model_config = ConfigDict(extra="allow")

    d_key: int | None = Field(default=None, ge=1, description="Key/query dim for global prior; defaults to d_hidden.")
    temperature: float = Field(default=1.0, gt=0.0, description="Softmax temperature for the global gate.")
    top_k: int | None = Field(default=None, ge=1, description="If set, keep only top‑k experts in the global prior.")
    use_sqrt_scale: bool = Field(default=True, description="Use 1/sqrt(d_key) (vs 1/d_key) dot‑product scaling.")
    init_scale_wq: float = Field(default=0.0, description="Uniform init scale for Wq; 0 → near‑uniform prior.")
    init_scale_wk: float = Field(default=0.0, description="Uniform init scale for Wk; 0 → near‑uniform prior.")

    d_key_local: int | None = Field(default=None, ge=1, description="Key dim for per‑token refiner; defaults to d_key.")
    local_temperature: float = Field(default=1.0, gt=0.0, description="Temperature for token‑refiner logits.")
    whisper_lambda: float = Field(default=0.1, ge=0.0, description="Strength λ of per-token refinement (0 disables).")
    center_refine: bool = Field(default=True, description="Center token logits over experts to redistribute mass only.")
    restrict_to_topk: bool = Field(default=True, description="Limit refinement to the global top‑k support if set.")


class ColumnBlockConfig(BlockConfig):
    """Column of experts with a shared router."""

    model_config = ConfigDict(extra="allow")

    block_type: str = "column"
    experts: list[SerializeAsAny[BlockConfig]]
    router: RouterConfig = Field(default_factory=RouterConfig, description="Router hyperparameters for this column.")
    alpha_init: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Initial scale for the shared ReZero gate α applied to BOTH the main MoE residual r_t "
            "and the correction head ρ(r_t): out = x + α·r_t + α·ρ(r_t). "
            "Smaller values keep the block near-identity at init; "
            "larger values engage both paths more strongly. Also scales gradient flow through both paths."
        ),
    )

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
    "PostUpGatedBlockConfig",
    "AdapterBlockConfig",
    "CortexStackConfig",
    "RouterConfig",
    "ColumnBlockConfig",
]
