from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from metta.agent.components.component_config import ComponentConfig


class MambaBackboneConfig(ComponentConfig):
    """Configuration for the MambaBackboneComponent."""

    in_key: str
    out_key: str
    name: str = "mamba_backbone"

    input_dim: int = 64
    d_model: int = 256
    d_intermediate: int = 512
    n_layer: int = 4
    ssm_cfg: Dict[str, Any] = Field(default_factory=dict)
    attn_layer_idx: List[int] = Field(default_factory=list)
    attn_cfg: Dict[str, Any] = Field(default_factory=dict)
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    dropout_p: float = 0.0
    max_cache_size: int = 128
    use_aux_tokens: bool = True
    last_action_dim: int = 1
    pool: Literal["cls", "mean", "none"] = "mean"
    ssm_expand: int = 2
    ssm_headdim: int = 16
    use_mem_eff_path: bool = True
    auto_align_stride: bool = True
    require_stride_multiple_of_eight: bool = True

    def make_component(self, env: Optional[Any] = None):  # type: ignore[override]
        from .backbone import MambaBackboneComponent

        return MambaBackboneComponent(config=self, env=env)

    @model_validator(mode="after")
    def _validate_attn_indices(self) -> "MambaBackboneConfig":
        for idx in self.attn_layer_idx:
            if idx < 0 or idx >= self.n_layer:
                raise ValueError(f"Attention layer index {idx} out of range for {self.n_layer} layers")
        return self

    def resolved_ssm_cfg(self) -> Dict[str, Any]:
        cfg = dict(self.ssm_cfg)
        cfg.setdefault("layer", "Mamba2")
        cfg.setdefault("expand", self.ssm_expand)
        cfg.setdefault("headdim", self.ssm_headdim)
        cfg.setdefault("use_mem_eff_path", self.use_mem_eff_path)

        expand = int(cfg["expand"])
        headdim = int(cfg["headdim"])
        if expand <= 0 or headdim <= 0:
            raise ValueError("expand and headdim must be positive integers")

        numerator = self.d_model * expand
        denominator = headdim
        divisible = numerator % denominator == 0
        if not divisible:
            if not self.auto_align_stride:
                raise ValueError(
                    f"d_model * expand must be divisible by headdim (got {self.d_model} * {expand} vs {headdim})"
                )
            multiplier = math.lcm(numerator, denominator) // numerator
            expand *= multiplier
            cfg["expand"] = expand
            numerator = self.d_model * expand

        ratio = numerator // denominator
        if self.require_stride_multiple_of_eight and ratio % 8 != 0:
            if not self.auto_align_stride:
                raise ValueError(
                    f"d_model * expand / headdim must be divisible by 8 (got ratio {ratio}). "
                    "Increase expand or adjust headdim."
                )
            align_multiplier = math.lcm(ratio, 8) // ratio
            expand *= align_multiplier
            cfg["expand"] = expand
            numerator = self.d_model * expand
            ratio = numerator // denominator

        if ratio <= 0:
            raise ValueError("Invalid resolved ratio for Mamba backbone.")

        cfg["expand"] = int(cfg["expand"])
        cfg["headdim"] = int(cfg["headdim"])
        return cfg
