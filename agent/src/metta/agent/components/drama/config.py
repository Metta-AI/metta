from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import model_validator

from metta.agent.components.component_config import ComponentConfig


@dataclass
class DramaMambaConfig:
    d_model: int = 512
    d_intermediate: int = 1024
    n_layer: int = 4
    stoch_dim: int = 128
    action_dim: int = 8
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)
    attn_layer_idx: list[int] = field(default_factory=list)
    attn_cfg: Dict[str, Any] = field(default_factory=dict)
    pff_cfg: Dict[str, Any] = field(default_factory=dict)
    dropout_p: float = 0.0
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


class DramaWorldModelConfig(ComponentConfig):
    """Config for Drama-inspired world model component."""

    name: str = "drama_world_model"
    in_key: str = "encoded_obs"
    out_key: str = "core"
    action_key: str = "last_actions"

    stoch_dim: int = 128
    action_dim: int = 8
    d_model: int = 512
    d_intermediate: int = 1024
    n_layer: int = 4
    dropout_p: float = 0.0

    use_reward_token: bool = True
    use_reset_token: bool = True
    pool: str = "mean"

    ssm_cfg: Dict[str, Any] = None
    attn_layer_idx: list[int] = None
    attn_cfg: Dict[str, Any] = None
    pff_cfg: Dict[str, Any] = None

    def make_component(self, env: Optional[Any] = None):  # type: ignore[override]
        from .world_model_component import DramaWorldModelComponent

        return DramaWorldModelComponent(config=self, env=env)

    @model_validator(mode="after")
    def _fill_defaults(self) -> "DramaWorldModelConfig":
        if self.ssm_cfg is None:
            self.ssm_cfg = {}
        if self.attn_layer_idx is None:
            self.attn_layer_idx = []
        if self.attn_cfg is None:
            self.attn_cfg = {}
        if self.pff_cfg is None:
            self.pff_cfg = {}
        return self
