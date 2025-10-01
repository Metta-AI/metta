"""Helpers for instantiating policies and loading checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cogames.policy import Policy, TrainablePolicy
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch


def instantiate_policy(class_path: str, env: Any, device: "torch.device") -> Policy:
    """Instantiate a policy class for the provided environment and device."""
    policy_class = load_symbol(class_path)
    return policy_class(env, device)


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a checkpoint path, descending into directories if needed."""
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidate_files = sorted((p for p in path.rglob("*.pt")), key=lambda target: target.stat().st_mtime)
    if not candidate_files:
        raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
    return candidate_files[-1]


def load_policy_checkpoint(policy: Policy, checkpoint: Optional[Path]) -> Optional[Path]:
    """Load policy weights if ``checkpoint`` is provided.

    Returns the resolved checkpoint path if loading occurred.
    """
    if checkpoint is None:
        return None

    resolved = resolve_checkpoint_path(checkpoint)
    if not isinstance(policy, TrainablePolicy):
        raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

    policy.load_policy_data(str(resolved))
    return resolved


__all__ = ["instantiate_policy", "resolve_checkpoint_path", "load_policy_checkpoint"]
