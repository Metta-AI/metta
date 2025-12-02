"""Configuration helpers for policy bindings and loss profiles."""

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from mettagrid.base_config import Config


class LossProfileConfig(Config):
    """Names the losses that should run for agents attached to this profile."""

    losses: List[str] = Field(default_factory=list)


class PolicyBindingConfig(Config):
    """Associates a policy loader with metadata used during rollout and training."""

    id: str = Field(description="Unique binding identifier")
    policy_uri: Optional[str] = Field(default=None, description="Checkpoint URI for neural policies")
    class_path: Optional[str] = Field(default=None, description="Import path for scripted policies")
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    trainable: bool = Field(default=True, description="Whether gradients should flow for this binding")
    loss_profile: Optional[str] = Field(default=None, description="Optional loss profile name for this binding")
    device: Optional[str] = Field(default=None, description="Optional device override for this binding")
    use_trainer_policy: bool = Field(
        default=False,
        description="If True, reuse the trainer-provided policy instance instead of loading a new one.",
    )

    @model_validator(mode="after")
    def validate_loader(self) -> "PolicyBindingConfig":
        if not self.use_trainer_policy and not (self.policy_uri or self.class_path):
            raise ValueError("policy_uri or class_path must be set unless use_trainer_policy=True")
        if self.use_trainer_policy and (self.policy_uri or self.class_path):
            raise ValueError("use_trainer_policy=True is mutually exclusive with policy_uri/class_path")
        return self
