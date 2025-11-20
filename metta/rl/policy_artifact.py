"""Backward-compatible shim for policy artifact utilities."""

from __future__ import annotations

from metta.agent.policy_artifact import *  # noqa: F401,F403

# Re-export primary API for existing importers
from metta.agent.policy_artifact import (  # noqa: F401
    PolicyArtifact,
    load_policy_artifact,
    policy_architecture_from_string,
    policy_architecture_to_string,
    save_policy_artifact_safetensors,
)

__all__ = [
    "PolicyArtifact",
    "load_policy_artifact",
    "policy_architecture_from_string",
    "policy_architecture_to_string",
    "save_policy_artifact_safetensors",
]
