from __future__ import annotations

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

    def make_component(self, env: Optional[Any] = None):  # type: ignore[override]
        from .backbone import MambaBackboneComponent

        return MambaBackboneComponent(config=self, env=env)

    @model_validator(mode="after")
    def _validate_attn_indices(self) -> "MambaBackboneConfig":
        for idx in self.attn_layer_idx:
            if idx < 0 or idx >= self.n_layer:
                raise ValueError(f"Attention layer index {idx} out of range for {self.n_layer} layers")
        return self
