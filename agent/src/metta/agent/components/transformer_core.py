"""Component configs for Transformer-style cores."""

from __future__ import annotations

from typing import Optional

from metta.agent.components.component_config import ComponentConfig

from .transformer_module import GTrXLModule, TransformerXLModule
from .transformer_nvidia_module import TransformerNvidiaCoreConfig  # re-export for convenience


class GTrXLCoreConfig(ComponentConfig):
    """Default GTrXL module used by the gtrxl policy variant."""

    name: str = "gtrxl_core"
    in_key: str = "encoded_obs"
    out_key: str = "core"

    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 6
    n_heads: int = 8
    d_ff: int = 512
    max_seq_len: int = 256
    memory_len: int = 0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    clamp_len: int = -1
    same_length: bool = False
    pre_lnorm: bool = True
    ext_len: int = 0

    def make_component(self, env: Optional[object] = None) -> GTrXLModule:
        return GTrXLModule(
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
            attn_type=0,
            use_gating=True,
            use_causal_mask=True,
        )


class TRXLCoreConfig(GTrXLCoreConfig):
    """Vanilla Transformer-XL core with memory."""

    latent_size: int = 256
    hidden_size: int = 256
    d_ff: int = 1024
    memory_len: int = 64

    def make_component(self, env: Optional[object] = None) -> TransformerXLModule:
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
            attn_type=0,
        )


# ---------------------------------------------------------------------------
# Backwards compatibility aliases (legacy names)
# ---------------------------------------------------------------------------

TransformerCoreConfig = GTrXLCoreConfig
TransformerImprovedCoreConfig = TRXLCoreConfig

__all__ = [
    "GTrXLCoreConfig",
    "TRXLCoreConfig",
    "TransformerCoreConfig",
    "TransformerImprovedCoreConfig",
    "TransformerNvidiaCoreConfig",
]
