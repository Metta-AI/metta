"""Optional helpers for working with metta policy artifacts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _policy_artifact_module() -> Any:
    """Load the optional metta RL policy artifact module."""
    try:
        return importlib.import_module("metta.rl.policy_artifact")
    except ImportError as e:
        raise ImportError(
            "Policy artifact support requires the metta RL extras; install metta[rl] to enable .mpt handling"
        ) from e


def load_policy_artifact(path: str | Path) -> Any:
    """Load a policy artifact from the given path."""
    module = _policy_artifact_module()
    return module.load_policy_artifact(Path(path))


def save_policy_artifact_safetensors(path: str | Path, policy_architecture: Any, state_dict: dict[str, Any]) -> None:
    """Save a policy artifact in safetensors format."""
    module = _policy_artifact_module()
    module.save_policy_artifact_safetensors(Path(path), policy_architecture=policy_architecture, state_dict=state_dict)
