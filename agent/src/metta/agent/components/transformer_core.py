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
        "latent_size": 64,
        "hidden_size": 64,
        "num_layers": 4,
        "n_heads": 4,
        "d_ff": 256,
        "max_seq_len": 256,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.05,
        "pre_lnorm": True,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
    },
    TransformerBackboneVariant.TRXL: {
        "latent_size": 48,
        "hidden_size": 64,
        "num_layers": 3,
        "n_heads": 4,
        "d_ff": 192,
        "max_seq_len": 192,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.05,
        "pre_lnorm": True,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
    },
    TransformerBackboneVariant.TRXL_NVIDIA: {
        "latent_size": 128,
        "hidden_size": 128,
        "num_layers": 6,
        "n_heads": 4,
        "d_ff": 512,
        "max_seq_len": 192,
        "memory_len": 32,
        "dropout": 0.05,
        "attn_dropout": 0.0,
        "pre_lnorm": False,
        "same_length": False,
        "clamp_len": -1,
        "positional_scale": 0.1,
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
        if self.variant is TransformerBackboneVariant.GTRXL:
            return GTrXLModule(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.num_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                memory_len=self.memory_len or 0,
                dropout=self.dropout,
                use_gating=True,
                use_causal_mask=True,
                positional_scale=self.positional_scale or 0.1,
            )

        if self.variant is TransformerBackboneVariant.TRXL:
            return TransformerXLModule(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.num_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                memory_len=self.memory_len or 0,
                dropout=self.dropout,
                dropatt=self.attn_dropout or 0.0,
                pre_lnorm=bool(self.pre_lnorm),
                same_length=bool(self.same_length),
                clamp_len=self.clamp_len or -1,
                ext_len=0,
                attn_type=0,
            )

        return NvidiaTransformerModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len or 0,
            dropout=self.dropout,
            dropatt=self.attn_dropout or 0.0,
            pre_lnorm=bool(self.pre_lnorm),
            clamp_len=self.clamp_len or -1,
        )


__all__ = ["TransformerBackboneConfig", "TransformerBackboneVariant"]
