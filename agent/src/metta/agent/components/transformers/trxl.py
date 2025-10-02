from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from metta.agent.components.transformer_module import TransformerXLModule


@dataclass
class TRXLConfig:
    """Backbone parameters for the Transformer-XL variant."""

    hidden_size: int = 32
    latent_size: int | None = None
    num_layers: int = 1
    n_heads: int = 2
    d_ff: int = 128
    max_seq_len: int = 192
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.05
    pre_lnorm: bool = True
    same_length: bool = False
    clamp_len: int = -1
    ext_len: int = 0
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    variant: str = "trxl"

    def build(self) -> TransformerXLModule:
        """Construct the Transformer-XL backbone module."""

        return TransformerXLModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            dropatt=self.attn_dropout,
            pre_lnorm=self.pre_lnorm,
            same_length=self.same_length,
            clamp_len=self.clamp_len,
            ext_len=self.ext_len,
            activation_checkpoint=self.activation_checkpoint,
            use_flash_checkpoint=self.use_flash_checkpoint,
            use_fused_layernorm=self.use_fused_layernorm,
            allow_tf32=self.allow_tf32,
        )

    def policy_defaults(self) -> Dict[str, Any]:
        """Return default policy-level overrides for this variant."""

        return {
            "manual_init": False,
            "strict_attr_indices": False,
            "learning_rate_hint": 9.0e-4,
        }


__all__ = ["TRXLConfig"]
