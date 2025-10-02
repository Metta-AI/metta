from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metta.agent.components.transformer_module import GTrXLModule

from .spec import TransformerSpec

if TYPE_CHECKING:  # pragma: no cover
    from metta.agent.components.transformer_core import TransformerBackboneConfig

DEFAULTS = {
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
}

POLICY_DEFAULTS = {
    "manual_init": False,
    "strict_attr_indices": False,
    "learning_rate_hint": 7.5e-4,
}


def build_backbone(config: "TransformerBackboneConfig", env: Any | None) -> GTrXLModule:
    memory_len = int(config.memory_len or 0)
    return GTrXLModule(
        d_model=config.hidden_size,
        n_heads=config.n_heads,
        n_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        memory_len=memory_len,
        dropout=config.dropout,
        use_gating=bool(config.use_gating),
        use_causal_mask=True,
        positional_scale=config.positional_scale or 0.1,
        attn_dropout=config.attn_dropout or config.dropout or 0.0,
        activation_checkpoint=bool(config.activation_checkpoint),
        use_flash_checkpoint=bool(config.use_flash_checkpoint),
        use_fused_layernorm=bool(config.use_fused_layernorm),
        allow_tf32=bool(config.allow_tf32),
    )


SPEC = TransformerSpec(
    defaults=DEFAULTS,
    policy_defaults=POLICY_DEFAULTS,
    builder=build_backbone,
)
