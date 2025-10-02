from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from metta.agent.components.sliding_transformer import SlidingTransformer, SlidingTransformerConfig


@dataclass
class SlidingTransformerBackboneConfig:
    """Parameters for the sliding-window transformer backbone."""

    in_key: str = "encoded_obs"
    out_key: str = "core"
    hidden_size: int = 16
    latent_size: int | None = None
    num_layers: int = 2
    n_heads: int = 1
    d_ff: int = 64
    max_cache_size: int = 80
    pool: Literal["cls", "mean", "none"] = "mean"

    def __post_init__(self) -> None:
        if self.latent_size is None:
            self.latent_size = self.hidden_size

    variant: str = "sliding"

    def build(self) -> SlidingTransformer:
        """Construct the sliding transformer backbone module."""

        hidden_size = self.hidden_size
        input_dim = self.latent_size or hidden_size
        ff_mult = max(1, self.d_ff // hidden_size)

        sliding_cfg = SlidingTransformerConfig(
            in_key=self.in_key,
            out_key=self.out_key,
            output_dim=hidden_size,
            input_dim=input_dim,
            num_heads=self.n_heads,
            ff_mult=ff_mult,
            num_layers=self.num_layers,
            max_cache_size=self.max_cache_size,
            pool=self.pool,
        )
        return SlidingTransformer(config=sliding_cfg, env=None)

    def policy_defaults(self) -> Dict[str, Any]:
        """Return default policy-level overrides for this variant."""

        return {
            "manual_init": False,
            "strict_attr_indices": False,
            "learning_rate_hint": 7.5e-4,
        }


__all__ = ["SlidingTransformerBackboneConfig"]
