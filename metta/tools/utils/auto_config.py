import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.collections import remove_falsey, remove_none_values
from metta.common.util.constants import METTA_AWS_ACCOUNT_ID
from metta.common.wandb.wandb_context import WandbConfig
from metta.setup.components.aws import AWSSetup
from metta.setup.components.observatory_key import ObservatoryKeySetup
from metta.setup.components.wandb import WandbSetup


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


def auto_wandb_config(run: str | None = None) -> WandbConfig:
    wandb_setup_module = WandbSetup()
    cfg = WandbConfig(
        **wandb_setup_module.to_config_settings(),  # type: ignore[arg-type]
        **supported_tool_overrides.to_config_settings(),
    )

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
        # If explicitly disabled, do not use stats server
        if self.STATS_SERVER_ENABLED is False:
            return {"stats_server_uri": None}
        # If explicitly provided, use stats server at given URI
        if self.STATS_SERVER_URI is not None:
            return {"stats_server_uri": self.STATS_SERVER_URI}
        return {}


supported_observatory_env_overrides = SupportedObservatoryEnvOverrides()


def auto_stats_server_uri() -> str | None:
    return {
        **ObservatoryKeySetup().to_config_settings(),  # type: ignore
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
    aws_setup_module = AWSSetup()
    return {
        **aws_setup_module.to_config_settings(),  # type: ignore
        **supported_aws_env_overrides.to_config_settings(),
    }.get("replay_dir")


def _join_prefix(prefix: str, run: str) -> str:
    cleaned_prefix = prefix.rstrip("/")
    return f"{cleaned_prefix}/{run}"


@dataclass(frozen=True)
class PolicyStorageDecision:
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
        return self.remote_prefix is not None


def auto_policy_storage_decision(run: str) -> PolicyStorageDecision:
    overrides = supported_aws_env_overrides.to_config_settings()
    override_prefix = overrides.get("policy_remote_prefix")
    if isinstance(override_prefix, str) and override_prefix:
        return PolicyStorageDecision(remote_prefix=_join_prefix(override_prefix, run), reason="env_override")

    aws_setup_module = AWSSetup()
    if not aws_setup_module.is_enabled():
        return PolicyStorageDecision(remote_prefix=None, reason="aws_not_enabled")

    aws_settings = aws_setup_module.to_config_settings()  # type: ignore[arg-type]
    base_prefix = aws_settings.get("policy_remote_prefix")
    if not isinstance(base_prefix, str) or not base_prefix:
        return PolicyStorageDecision(remote_prefix=None, reason="no_base_prefix")

    connected_account = aws_setup_module.check_connected_as()
    if connected_account != METTA_AWS_ACCOUNT_ID:
        return PolicyStorageDecision(remote_prefix=None, reason="not_connected")

    return PolicyStorageDecision(remote_prefix=_join_prefix(base_prefix, run), reason="softmax_connected")


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
