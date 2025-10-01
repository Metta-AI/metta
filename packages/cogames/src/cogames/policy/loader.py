"""Helpers for instantiating policies and loading checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cogames.policy import Policy, TrainablePolicy
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch


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


def load_policy_checkpoint(policy: Policy, checkpoint: str) -> Path:
    """Load policy weights from ``checkpoint``

    Returns the resolved checkpoint path.
    """
    resolved = resolve_checkpoint_path(Path(checkpoint))
    if not isinstance(policy, TrainablePolicy):
        raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

    policy.load_policy_data(str(resolved))
    return resolved


def instantiate_or_load_policy(
    policy_class_path: str, policy_data_path: Optional[str], env: Any, device: "torch.device | None" = None
) -> Policy:
    import torch

    policy_class = load_symbol(policy_class_path)
    policy = policy_class(env, device or torch.device("cpu"))

    if policy_data_path:
        load_policy_checkpoint(policy, policy_data_path)
    return policy


__all__ = ["instantiate_or_load_policy"]
