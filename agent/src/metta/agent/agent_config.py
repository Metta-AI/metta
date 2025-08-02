from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra


class PolicySelectorConfig(BaseModelWithForbidExtra):
    uri: str | None = None
    type: Literal["top", "latest", "specific"] = "top"
    range: int = Field(default=0, ge=0)
    metric: str = "final.score"
    generation: int | None = None


class ObservationsConfig(BaseModelWithForbidExtra):
    obs_key: str = Field(default="grid_obs")


class AgentConfig(BaseModelWithForbidExtra):
    target_: str = Field(default="metta.agent.metta_agent.MettaAgent", alias="_target_")

    # Optional policy selector
    policy_selector: PolicySelectorConfig | None = None

    # Observations configuration
    observations: ObservationsConfig = Field(default_factory=ObservationsConfig)

    # Weight clipping and analysis
    clip_range: float = Field(default=0, ge=0)
    analyze_weights_interval: int = Field(default=300, ge=0)
    l2_init_weight_update_interval: int = Field(default=0, ge=0)

    # Components - stored as a dict to maintain compatibility
    components: dict[str, Any]

    @property
    def _target_(self) -> str:
        """Provide backward compatibility for accessing _target_."""
        return self.target_

    @model_validator(mode="after")
    def validate_required_components(self) -> "AgentConfig":
        required = ["_obs_", "_core_", "_action_embeds_", "_action_", "_value_"]
        for comp in required:
            if comp not in self.components:
                raise ValueError(f"Missing required component: {comp}")
        return self

    @model_validator(mode="after")
    def validate_core_fields(self) -> "AgentConfig":
        core = self.components.get("_core_", {})
        if "output_size" not in core:
            raise ValueError("_core_ component must have 'output_size' defined")
        if "nn_params" not in core or "num_layers" not in core.get("nn_params", {}):
            raise ValueError("_core_ component must have 'nn_params.num_layers' defined")
        return self


def create_agent_config(cfg: DictConfig) -> AgentConfig:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("agent config must be a dict")
    return AgentConfig.model_validate(config_dict)
