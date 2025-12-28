from __future__ import annotations

from typing import Mapping

from pydantic import model_validator

from metta.agent.policy import Policy, PolicyArchitecture
from mettagrid.base_config import Config


class PolicyAssetConfig(Config):
    """Declarative spec for a policy asset.

    A policy asset can either be loaded from a URI, created from an architecture, or both
    (load if available, otherwise create).
    """

    uri: str | None = None
    architecture: PolicyArchitecture | None = None

    # Whether this policy should be checkpointed during training.
    checkpoint: bool = True

    # Whether this policy is trainable (affects parameter freezing; training loop still trains the primary policy).
    trainable: bool = True

    @model_validator(mode="after")
    def _validate_asset(self) -> "PolicyAssetConfig":
        if self.uri is None and self.architecture is None:
            raise ValueError("PolicyAssetConfig must provide at least one of: uri, architecture")
        return self


class PolicyAssetRegistry:
    """Runtime registry of named policy assets."""

    def __init__(
        self,
        *,
        configs: Mapping[str, PolicyAssetConfig],
        policies: Mapping[str, Policy],
        primary: str,
    ) -> None:
        self._configs = dict(configs)
        self._policies = dict(policies)
        self._primary = primary

        if primary not in self._policies:
            raise KeyError(f"Primary policy '{primary}' is missing from loaded policies: {list(self._policies)}")

    @property
    def primary_name(self) -> str:
        return self._primary

    @property
    def primary_policy(self) -> Policy:
        return self._policies[self._primary]

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


