from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from metta.agent.components.transformer_nvidia_module import NvidiaTransformerModule


@dataclass
class TRXLNvidiaConfig:
    """Backbone parameters for the NVIDIA-optimized Transformer-XL variant."""

    hidden_size: int = 48
    latent_size: int | None = None
    num_layers: int = 2
    n_heads: int = 2
    d_ff: int = 192
    max_seq_len: int = 192
    memory_len: int = 32
    dropout: float = 0.05
    attn_dropout: float = 0.0
    pre_lnorm: bool = False
    clamp_len: int = -1
    ext_len: int = 0
    activation_checkpoint: bool = False
    use_flash_checkpoint: bool = False
    allow_tf32: bool = True
    use_fused_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    variant: str = "trxl_nvidia"

    def build(self) -> NvidiaTransformerModule:
        """Construct the NVIDIA Transformer-XL backbone module."""

        return NvidiaTransformerModule(
            d_model=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            memory_len=self.memory_len,
            dropout=self.dropout,
            dropatt=self.attn_dropout,
            pre_lnorm=self.pre_lnorm,
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
            "manual_init": True,
            "strict_attr_indices": True,
            "learning_rate_hint": 3.0e-4,
        }


__all__ = ["TRXLNvidiaConfig"]
