"""Shared configuration helpers used across metta packages."""

from importlib import import_module
from typing import Callable

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
    register_setup_factories,
)


def _lazy_setup_factory(module_path: str, class_name: str) -> Callable[[], object]:
    def _factory() -> object:
        module = import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    return _factory


register_setup_factories(
    aws_factory=_lazy_setup_factory("metta.setup.components.aws", "AWSSetup"),
    observatory_factory=_lazy_setup_factory("metta.setup.components.observatory_key", "ObservatoryKeySetup"),
    wandb_factory=_lazy_setup_factory("metta.setup.components.wandb", "WandbSetup"),
)


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
