"""Unified configuration for transformer policy backbones."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from metta.agent.components.component_config import ComponentConfig

from .transformers import available_backbones, get_backbone_spec


class TransformerBackboneConfig(ComponentConfig):
    """Configures the transformer stack used by transformer policies."""

    name: str = "transformer_backbone"
    in_key: str = "encoded_obs"
    out_key: str = "core"

    variant: str = "gtrxl"

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
    max_cache_size: int | None = None
    pool: Literal["cls", "mean", "none"] | None = None

    @model_validator(mode="after")
    def _apply_variant_defaults(self) -> "TransformerBackboneConfig":
        spec = get_backbone_spec(self.variant)
        for field_name, default_value in spec.defaults.items():
            if getattr(self, field_name, None) is None:
                setattr(self, field_name, default_value)

        if self.latent_size is None and self.hidden_size is not None:
            self.latent_size = self.hidden_size

        return self

    # ------------------------------------------------------------------
    # ComponentConfig API
    # ------------------------------------------------------------------

    def make_component(self, env: Any | None = None):  # type: ignore[override]
        spec = get_backbone_spec(self.variant)
        return spec.builder(self, env)


__all__ = ["TransformerBackboneConfig", "available_backbones"]
