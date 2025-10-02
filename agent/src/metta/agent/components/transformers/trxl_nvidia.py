from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metta.agent.components.transformer_nvidia_module import NvidiaTransformerModule

from .spec import TransformerSpec

if TYPE_CHECKING:  # pragma: no cover
    from metta.agent.components.transformer_core import TransformerBackboneConfig

DEFAULTS = {
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
}

POLICY_DEFAULTS = {
    "manual_init": True,
    "strict_attr_indices": True,
    "learning_rate_hint": 3.0e-4,
}


def build_backbone(config: "TransformerBackboneConfig", env: Any | None) -> NvidiaTransformerModule:
    memory_len = int(config.memory_len or 0)
    return NvidiaTransformerModule(
        d_model=config.hidden_size,
        n_heads=config.n_heads,
        n_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        memory_len=memory_len,
        dropout=config.dropout,
        dropatt=config.attn_dropout or 0.0,
        pre_lnorm=bool(config.pre_lnorm),
        clamp_len=config.clamp_len or -1,
        ext_len=int(config.ext_len or 0),
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
