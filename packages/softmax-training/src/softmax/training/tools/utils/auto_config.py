"""Backward-compatible wrapper for configuration helpers.

The auto-configuration utilities now live in :mod:`softmax.config.auto_config`.
Import from there directly in new code so we can retire this wrapper when the
package split lands.
"""

from __future__ import annotations

from warnings import warn

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

warn(
    "softmax.training.tools.utils.auto_config is deprecated; import from softmax.config.auto_config instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
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
