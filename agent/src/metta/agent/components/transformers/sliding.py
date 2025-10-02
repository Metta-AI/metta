from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metta.agent.components.sliding_transformer import SlidingTransformer, SlidingTransformerConfig

from .spec import TransformerSpec

if TYPE_CHECKING:  # pragma: no cover
    from metta.agent.components.transformer_core import TransformerBackboneConfig

DEFAULTS = {
    "latent_size": 16,
    "hidden_size": 16,
    "num_layers": 2,
    "n_heads": 1,
    "d_ff": 64,
    "max_seq_len": 80,
    "memory_len": 0,
    "dropout": 0.0,
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
    "max_cache_size": 80,
    "pool": "mean",
}

POLICY_DEFAULTS = {
    "manual_init": False,
    "strict_attr_indices": False,
    "learning_rate_hint": 7.5e-4,
}


def build_backbone(config: "TransformerBackboneConfig", env: Any | None) -> SlidingTransformer:
    hidden_size = config.hidden_size or config.latent_size or DEFAULTS["hidden_size"]
    input_dim = config.latent_size or hidden_size
    ff_mult = max(1, (config.d_ff or hidden_size * 4) // hidden_size)

    sliding_cfg = SlidingTransformerConfig(
        in_key=config.in_key,
        out_key=config.out_key,
        output_dim=hidden_size,
        input_dim=input_dim,
        num_heads=config.n_heads or DEFAULTS["n_heads"],
        ff_mult=ff_mult,
        num_layers=config.num_layers or DEFAULTS["num_layers"],
        max_cache_size=config.max_cache_size or DEFAULTS["max_cache_size"],
        pool=config.pool or DEFAULTS["pool"],
    )
    return SlidingTransformer(config=sliding_cfg, env=env)


SPEC = TransformerSpec(
    defaults=DEFAULTS,
    policy_defaults=POLICY_DEFAULTS,
    builder=build_backbone,
)
