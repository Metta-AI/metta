from __future__ import annotations

from typing import Literal, Mapping

from pydantic import Field, model_validator

from metta.agent.policy import Policy, PolicyArchitecture
from mettagrid.base_config import Config


class OptimizerConfig(Config):
    type: Literal["adam", "muon", "adamw_schedulefree", "sgd_schedulefree"] = "adamw_schedulefree"
    # Learning rate tuned from CvC sweep winners (schedule-free AdamW)
    learning_rate: float = Field(default=0.00737503357231617, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon tuned from CvC sweep winners
    eps: float = Field(default=5.0833278919526e-07, gt=0)
    # Weight decay: modest L2 regularization for AdamW-style optimizers
    weight_decay: float = Field(default=0.01, ge=0)
    # ScheduleFree-specific parameters
    momentum: float = Field(default=0.9, ge=0, le=1.0)  # Beta parameter for ScheduleFree
    warmup_steps: int = Field(default=1000, ge=0)  # Number of warmup steps for ScheduleFree


class PolicyAssetConfig(Config):
    """Declarative spec for a policy asset.

    A policy asset can either be loaded from a URI, created from an architecture, or both
    (load if available, otherwise create).
    """

    uri: str | None = None
    architecture: PolicyArchitecture | None = None

    # Whether this policy should be checkpointed during training.
    checkpoint: bool = True

    # Whether this policy is trainable (affects parameter freezing and optimizer creation).
    trainable: bool = True

    # Optimizer configuration for this policy. Defaults to the standard optimizer config.
    # Set to None for frozen/non-trainable policies or to disable optimization.
    optimizer: OptimizerConfig | None = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def _validate_asset(self) -> "PolicyAssetConfig":
        if self.uri is None and self.architecture is None:
            raise ValueError("PolicyAssetConfig must provide at least one of: uri, architecture")
        if not self.trainable:
            self.optimizer = None
        return self


class PolicyAssetRegistry:
    """Runtime registry of named policy assets."""

    def __init__(
        self,
        *,
        configs: Mapping[str, PolicyAssetConfig],
        policies: Mapping[str, Policy],
    ) -> None:
        self._configs = dict(configs)
        self._policies = dict(policies)

    @property
    def policies(self) -> dict[str, Policy]:
        return self._policies

    @property
    def configs(self) -> dict[str, PolicyAssetConfig]:
        return self._configs

    def get(self, name: str) -> Policy:
        return self._policies[name]

    def get_config(self, name: str) -> PolicyAssetConfig:
        return self._configs[name]
