"""Shared configuration helpers used across metta packages."""

from softmax.config import auto_config
from softmax.config.auto_config import (
    PolicyStorageDecision,
    SupportedAwsEnvOverrides,
    SupportedObservatoryEnvOverrides,
    SupportedWandbEnvOverrides,
    auto_policy_storage_decision,
    auto_replay_dir,
    auto_run_name,
    auto_stats_server_uri,
    auto_wandb_config,
)
from softmax.config.bootstrap import ensure_setup_factories_registered

ensure_setup_factories_registered()


__all__ = [
    "auto_config",
    "PolicyStorageDecision",
    "SupportedAwsEnvOverrides",
    "SupportedObservatoryEnvOverrides",
    "SupportedWandbEnvOverrides",
    "auto_policy_storage_decision",
    "auto_replay_dir",
    "auto_run_name",
    "auto_stats_server_uri",
    "auto_wandb_config",
]
