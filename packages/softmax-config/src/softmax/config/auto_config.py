"""Shared auto-configuration helpers for tooling and training flows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.collections import remove_falsey, remove_none_values
from metta.common.util.constants import METTA_AWS_ACCOUNT_ID
from metta.common.wandb.context import WandbConfig


class _SetupModule(Protocol):
    def to_config_settings(self) -> dict[str, Any]: ...


@runtime_checkable
class _AwsSetupModule(_SetupModule, Protocol):
    def is_enabled(self) -> bool: ...

    def check_connected_as(self) -> str: ...


WandbSetupFactoryType = Callable[[], _SetupModule]
ObservatorySetupFactoryType = Callable[[], _SetupModule]
AwsSetupFactoryType = Callable[[], _AwsSetupModule]


@dataclass
class _SetupFactories:
    aws_factory: AwsSetupFactoryType | None = None
    observatory_factory: ObservatorySetupFactoryType | None = None
    wandb_factory: WandbSetupFactoryType | None = None


_factories = _SetupFactories()

AWSSetup: AwsSetupFactoryType | None = None
ObservatoryKeySetup: ObservatorySetupFactoryType | None = None
WandbSetup: WandbSetupFactoryType | None = None


def register_setup_factories(
    *,
    aws_factory: AwsSetupFactoryType | None = None,
    observatory_factory: ObservatorySetupFactoryType | None = None,
    wandb_factory: WandbSetupFactoryType | None = None,
) -> None:
    """Register callables that construct setup components."""

    global AWSSetup, ObservatoryKeySetup, WandbSetup

    if aws_factory is not None:
        _factories.aws_factory = aws_factory
        AWSSetup = aws_factory
    if observatory_factory is not None:
        _factories.observatory_factory = observatory_factory
        ObservatoryKeySetup = observatory_factory
    if wandb_factory is not None:
        _factories.wandb_factory = wandb_factory
        WandbSetup = wandb_factory


def reset_setup_factories() -> None:
    """Reset registered factories; primarily used by tests."""

    global AWSSetup, ObservatoryKeySetup, WandbSetup

    _factories.aws_factory = None
    _factories.observatory_factory = None
    _factories.wandb_factory = None
    AWSSetup = None
    ObservatoryKeySetup = None
    WandbSetup = None


def _require_factory(
    name: str,
    factory: Callable[[], _SetupModule] | None,
) -> Callable[[], _SetupModule]:
    if factory is None:
        raise RuntimeError(
            f"Setup factory '{name}' is not registered. Call "
            "softmax.config.auto_config.register_setup_factories(...) during startup."
        )
    return factory


def _wandb_setup() -> _SetupModule:
    factory = _require_factory("wandb", _factories.wandb_factory)
    return factory()


def _observatory_setup() -> _SetupModule:
    factory = _require_factory("observatory", _factories.observatory_factory)
    return factory()


def _aws_setup() -> _AwsSetupModule:
    factory = _require_factory("aws", _factories.aws_factory)
    module = factory()
    if not isinstance(module, _AwsSetupModule):
        raise TypeError("AWS setup factory returned an incompatible object")
    return module


class SupportedWandbEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    WANDB_ENABLED: bool | None = Field(default=None, description="Enable Weights & Biases")
    WANDB_PROJECT: str | None = Field(default=None, description="Weights & Biases project")
    WANDB_ENTITY: str | None = Field(default=None, description="Weights & Biases entity")

    def to_config_settings(self) -> dict[str, str | bool]:
        return remove_none_values(
            {
                "enabled": self.WANDB_ENABLED,
                "project": self.WANDB_PROJECT,
                "entity": self.WANDB_ENTITY,
            }
        )


supported_tool_overrides = SupportedWandbEnvOverrides()


def _merge_wandb_settings(*settings_dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for settings in settings_dicts:
        for key, value in settings.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def auto_wandb_config(run: str | None = None) -> WandbConfig:
    wandb_setup_module = _wandb_setup()
    merged_settings = _merge_wandb_settings(
        WandbConfig.Off().model_dump(),
        wandb_setup_module.to_config_settings(),
        supported_tool_overrides.to_config_settings(),
    )

    cfg = WandbConfig(**merged_settings)

    if run:
        cfg.run_id = run
        cfg.group = run
        cfg.data_dir = f"./train_dir/{run}"

    return cfg


class SupportedObservatoryEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    STATS_SERVER_ENABLED: bool | None = Field(default=None, description="If true, use the stats server")
    STATS_SERVER_URI: str | None = Field(default=None, description="Stats server URI")

    def to_config_settings(self) -> dict[str, str | None]:
        if self.STATS_SERVER_ENABLED is False:
            return {"stats_server_uri": None}
        if self.STATS_SERVER_URI is not None:
            return {"stats_server_uri": self.STATS_SERVER_URI}
        return {}


supported_observatory_env_overrides = SupportedObservatoryEnvOverrides()


def auto_stats_server_uri() -> str | None:
    return {
        **_observatory_setup().to_config_settings(),
        **supported_observatory_env_overrides.to_config_settings(),
    }.get("stats_server_uri")


class SupportedAwsEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    REPLAY_DIR: str | None = Field(default=None, description="Replay directory")
    POLICY_REMOTE_PREFIX: str | None = Field(default=None, description="Override policy remote prefix (s3://...)")

    def to_config_settings(self) -> dict[str, str]:
        return remove_none_values(
            {
                "replay_dir": self.REPLAY_DIR,
                "policy_remote_prefix": self.POLICY_REMOTE_PREFIX,
            }
        )


supported_aws_env_overrides = SupportedAwsEnvOverrides()


def auto_replay_dir() -> str:
    aws_setup_module = _aws_setup()
    return {
        **aws_setup_module.to_config_settings(),
        **supported_aws_env_overrides.to_config_settings(),
    }.get("replay_dir")


def _join_prefix(prefix: str, run: str | None) -> str:
    if run is None:
        return prefix.rstrip("/")
    cleaned_prefix = prefix.rstrip("/")
    return f"{cleaned_prefix}/{run}"


@dataclass(frozen=True)
class PolicyStorageDecision:
    base_prefix: str | None
    remote_prefix: str | None
    reason: Literal[
        "env_override",
        "softmax_connected",
        "aws_not_enabled",
        "no_base_prefix",
        "not_connected",
    ]

    @property
    def using_remote(self) -> bool:
        return self.base_prefix is not None and self.reason in {"env_override", "softmax_connected"}


def auto_policy_storage_decision(run: str | None = None) -> PolicyStorageDecision:
    overrides = supported_aws_env_overrides.to_config_settings()
    override_prefix = overrides.get("policy_remote_prefix")
    if isinstance(override_prefix, str) and override_prefix:
        cleaned = override_prefix.rstrip("/")
        remote = _join_prefix(cleaned, run) if run else None
        return PolicyStorageDecision(base_prefix=cleaned, remote_prefix=remote, reason="env_override")

    aws_setup_module = _aws_setup()
    if not aws_setup_module.is_enabled():
        return PolicyStorageDecision(base_prefix=None, remote_prefix=None, reason="aws_not_enabled")

    aws_settings = aws_setup_module.to_config_settings()
    base_prefix = aws_settings.get("policy_remote_prefix")
    if not isinstance(base_prefix, str) or not base_prefix:
        return PolicyStorageDecision(base_prefix=None, remote_prefix=None, reason="no_base_prefix")
    cleaned_base = base_prefix.rstrip("/")

    connected_account = aws_setup_module.check_connected_as()
    if connected_account != METTA_AWS_ACCOUNT_ID:
        return PolicyStorageDecision(base_prefix=cleaned_base, remote_prefix=None, reason="not_connected")

    remote = _join_prefix(cleaned_base, run) if run else None
    return PolicyStorageDecision(base_prefix=cleaned_base, remote_prefix=remote, reason="softmax_connected")


def auto_run_name(prefix: str | None = None) -> str:
    return ".".join(
        remove_falsey(
            [
                prefix,
                os.getenv("USER", "unknown"),
                datetime.now().strftime("%Y%m%d.%H%M%S"),
            ]
        )
    )


__all__ = [
    "AWSSetup",
    "AwsSetupFactoryType",
    "ObservatoryKeySetup",
    "ObservatorySetupFactoryType",
    "WandbSetup",
    "WandbSetupFactoryType",
    "PolicyStorageDecision",
    "SupportedAwsEnvOverrides",
    "SupportedObservatoryEnvOverrides",
    "SupportedWandbEnvOverrides",
    "auto_policy_storage_decision",
    "auto_replay_dir",
    "auto_run_name",
    "auto_stats_server_uri",
    "auto_wandb_config",
    "register_setup_factories",
    "reset_setup_factories",
]
