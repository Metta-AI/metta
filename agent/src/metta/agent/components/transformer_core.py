"""Unified configuration for transformer policy backbones."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from pydantic import Field, model_validator

from metta.agent.components.component_config import ComponentConfig

from .transformer_module import GTrXLModule, TransformerXLModule
from .transformer_nvidia_module import NvidiaTransformerModule


class TransformerBackboneVariant(str, Enum):
    """Supported transformer backbone variants."""

    GTRXL = "gtrxl"
    TRXL = "trxl"
    TRXL_NVIDIA = "trxl_nvidia"


_VARIANT_DEFAULTS: Dict[TransformerBackboneVariant, Dict[str, Any]] = {
    TransformerBackboneVariant.GTRXL: {
        "latent_size": 32,
        "hidden_size": 32,
        "num_layers": 1,
        "n_heads": 2,
        "d_ff": 128,
        "max_seq_len": 256,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.05,
        "pre_lnorm": True,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
        "use_gating": True,
        "ext_len": 0,
        "activation_checkpoint": False,
        "use_flash_checkpoint": False,
        "allow_tf32": True,
        "use_fused_layernorm": False,
    },
    TransformerBackboneVariant.TRXL: {
        "latent_size": 32,
        "hidden_size": 32,
        "num_layers": 1,
        "n_heads": 2,
        "d_ff": 128,
        "max_seq_len": 192,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.05,
        "pre_lnorm": True,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
        "use_gating": False,
        "ext_len": 0,
        "activation_checkpoint": False,
        "use_flash_checkpoint": False,
        "allow_tf32": True,
        "use_fused_layernorm": False,
    },
    TransformerBackboneVariant.TRXL_NVIDIA: {
        "latent_size": 48,
        "hidden_size": 48,
        "num_layers": 2,
        "n_heads": 2,
        "d_ff": 192,
        "max_seq_len": 192,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.0,
        "pre_lnorm": False,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
        "use_gating": False,
        "ext_len": 0,
        "activation_checkpoint": False,
        "use_flash_checkpoint": False,
        "allow_tf32": True,
        "use_fused_layernorm": False,
    },
}


class TransformerBackboneConfig(ComponentConfig):
    """Configures the transformer stack used by transformer policies."""

    name: str = "transformer_backbone"
    in_key: str = "encoded_obs"
    out_key: str = "core"

    variant: TransformerBackboneVariant = TransformerBackboneVariant.GTRXL

    latent_size: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None
    max_seq_len: int | None = None
    memory_len: int | None = None
    dropout: float | None = None
    attn_dropout: float | None = None
    pre_lnorm: bool | None = None
    same_length: bool | None = None
    clamp_len: int | None = None
    positional_scale: float | None = Field(default=None, description="Scaling applied to sinusoidal embeddings")
    use_gating: bool | None = None
    ext_len: int | None = None
    activation_checkpoint: bool | None = None
    use_flash_checkpoint: bool | None = None
    allow_tf32: bool | None = None
    use_fused_layernorm: bool | None = None

    @model_validator(mode="after")
    def _apply_variant_defaults(self) -> "TransformerBackboneConfig":
        defaults = _VARIANT_DEFAULTS[self.variant]
        for field_name, default_value in defaults.items():
            if getattr(self, field_name) is None:
                setattr(self, field_name, default_value)

        if self.latent_size is None and self.hidden_size is not None:
            self.latent_size = self.hidden_size

        return self

    # ------------------------------------------------------------------
    # ComponentConfig API
    # ------------------------------------------------------------------

    def make_component(self, env: Any | None = None):  # type: ignore[override]
        memory_len = int(self.memory_len or 0)

        if self.variant is TransformerBackboneVariant.GTRXL:
            core = GTrXLModule(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.num_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                memory_len=memory_len,
                dropout=self.dropout,
                use_gating=bool(self.use_gating),
                use_causal_mask=True,
                positional_scale=self.positional_scale or 0.1,
                attn_dropout=self.attn_dropout or self.dropout or 0.0,
                activation_checkpoint=bool(self.activation_checkpoint),
                use_flash_checkpoint=bool(self.use_flash_checkpoint),
                use_fused_layernorm=bool(self.use_fused_layernorm),
            )
        elif self.variant is TransformerBackboneVariant.TRXL:
            core = TransformerXLModule(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.num_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                memory_len=memory_len,
                dropout=self.dropout,
                dropatt=self.attn_dropout or 0.0,
                pre_lnorm=bool(self.pre_lnorm),
                same_length=bool(self.same_length),
                clamp_len=self.clamp_len or -1,
                ext_len=int(self.ext_len or 0),
                attn_type=0,
                activation_checkpoint=bool(self.activation_checkpoint),
                use_flash_checkpoint=bool(self.use_flash_checkpoint),
                use_fused_layernorm=bool(self.use_fused_layernorm),
            )
        else:
            core = NvidiaTransformerModule(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.num_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                memory_len=memory_len,
                dropout=self.dropout,
                dropatt=self.attn_dropout or 0.0,
                pre_lnorm=bool(self.pre_lnorm),
                clamp_len=self.clamp_len or -1,
                ext_len=int(self.ext_len or 0),
                activation_checkpoint=bool(self.activation_checkpoint),
                use_flash_checkpoint=bool(self.use_flash_checkpoint),
                use_fused_layernorm=bool(self.use_fused_layernorm),
            )

        return core


__all__ = ["TransformerBackboneConfig", "TransformerBackboneVariant"]
