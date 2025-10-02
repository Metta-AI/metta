from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from metta.agent.components.transformer_module import GTrXLModule


@dataclass
class GTrXLConfig:
    """Backbone parameters for the GTrXL transformer."""

    hidden_size: int = 32
    latent_size: int | None = None
    num_layers: int = 1
    n_heads: int = 2
    d_ff: int = 128
    max_seq_len: int = 256
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.05
    positional_scale: float = 0.1
    use_gating: bool = True
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    variant: str = "gtrxl"

    def build(self) -> GTrXLModule:
        """Construct the GTrXL backbone module."""

        return GTrXLModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            use_gating=self.use_gating,
            positional_scale=self.positional_scale,
            attn_dropout=self.attn_dropout,
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
            "learning_rate_hint": 7.5e-4,
        }


__all__ = ["GTrXLConfig"]
