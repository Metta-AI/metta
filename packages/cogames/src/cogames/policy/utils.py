"""Policy helper utilities, including LSTM state adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from cogames.policy.interfaces import Policy, TrainablePolicy
from mettagrid.util.module import load_symbol

_POLICY_CLASS_SHORTHAND: dict[str, str] = {
    "random": "cogames.policy.random.RandomPolicy",
    "noop": "cogames.policy.noop.NoopPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "token": "cogames.policy.token.TokenPolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
    "claude": "cogames.policy.claude.ClaudePolicy",
}


def initialize_or_load_policy(
    policy_class_path: str, policy_data_path: Optional[str], env: Any, device: "torch.device | None" = None
) -> Policy:
    policy_class = load_symbol(policy_class_path)
    policy = policy_class(env, device or torch.device("cpu"))

    if policy_data_path:
        if not isinstance(policy, TrainablePolicy):
            raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

        policy.load_policy_data(policy_data_path)
    return policy


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "simple", "token", "lstm" or a full class path.

    Returns:
        Full class path to the policy.
    """
    full_path = _POLICY_CLASS_SHORTHAND.get(policy, policy)

    # Will raise an error if invalid
    _ = load_symbol(full_path)
    return full_path


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    return {v: k for k, v in _POLICY_CLASS_SHORTHAND.items()}.get(policy)


_NOT_CHECKPOINT_PATTERNS = (
    r"trainer_state\.pt",  # trainer state file
    r"model_\d{6}\.pt",  # matches model_000001.pt etc
)


def find_policy_checkpoints(checkpoints_path: Path, env_name: Optional[str] = None) -> list[Path]:
    checkpoints = []
    if env_name:
        # Try to find the final checkpoint
        # PufferLib saves checkpoints in data_dir/env_name/
        checkpoint_dir = checkpoints_path / env_name
        if checkpoint_dir.exists():
            checkpoints = checkpoint_dir.glob("*.pt")

    # Fallback: also check directly in checkpoints_path
    if not checkpoints and checkpoints_path.exists():
        checkpoints = checkpoints_path.glob("*.pt")
    return [
        p
        for p in sorted(checkpoints, key=lambda c: c.stat().st_mtime)
        if not any(re.fullmatch(pattern, p.name) for pattern in _NOT_CHECKPOINT_PATTERNS)
    ]


def resolve_policy_data_path(
    policy_data_path: Optional[str],
) -> Optional[str]:
    """Resolve a checkpoint path if provided.

    If the supplied path does not exist locally and AWS policy storage is configured,
    this will attempt to download the checkpoint into the requested location.
    """

    if policy_data_path is None:
        return None

    path = Path(policy_data_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        checkpoints = find_policy_checkpoints(path)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
        return str(checkpoints[-1])

    if path.exists():  # Non-pt extension but present
        return str(path)

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


LSTMStateTuple = Tuple[torch.Tensor, torch.Tensor]
LSTMStateDict = Dict[str, torch.Tensor]


def _canonical_component(component: torch.Tensor, expected_layers: Optional[int]) -> torch.Tensor:
    """Return a ``(layers, batch, hidden)`` tensor, adding axes as needed."""
    if component.dim() > 3:
        msg = f"Expected tensor with <=3 dims, got {component.dim()}"
        raise ValueError(msg)

    while component.dim() < 3:
        component = component.unsqueeze(0)

    if expected_layers is not None:
        if component.shape[0] != expected_layers and component.shape[1] == expected_layers:
            component = component.transpose(0, 1)
        if component.shape[0] != expected_layers:
            msg = f"Hidden state has unexpected layer dimension. Expected {expected_layers}, got {component.shape[0]}."
            raise ValueError(msg)

    return component.contiguous()


@dataclass
class LSTMState:
    """Canonical representation of an LSTM hidden/cell state."""

    hidden: torch.Tensor
    cell: torch.Tensor

    @classmethod
    def from_tuple(
        cls,
        state: Optional[LSTMStateTuple],
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if state is None:
            return None
        hidden, cell = state
        return cls(
            _canonical_component(hidden, expected_layers),
            _canonical_component(cell, expected_layers),
        )

    @classmethod
    def from_dict(
        cls,
        state: LSTMStateDict,
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if not state:
            return None
        hidden = state.get("lstm_h")
        cell = state.get("lstm_c")
        if hidden is None or cell is None:
            return None
        return cls(
            _canonical_component(hidden, expected_layers),
            _canonical_component(cell, expected_layers),
        )

    @classmethod
    def from_any(
        cls,
        state: Optional[Union["LSTMState", LSTMStateTuple, LSTMStateDict]],
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if state is None:
            return None
        if isinstance(state, LSTMState):
            return state
        if isinstance(state, dict):
            return cls.from_dict(state, expected_layers)
        if isinstance(state, tuple):
            return cls.from_tuple(state, expected_layers)
        msg = f"Unsupported LSTM state container type: {type(state)!r}"
        raise TypeError(msg)

    def to_tuple(self) -> LSTMStateTuple:
        return self.hidden, self.cell

    def write_dict(self, target: LSTMStateDict) -> None:
        """Populate ``target`` with tensors in batch-major form."""
        target.clear()
        target["lstm_h"] = self.hidden.transpose(0, 1).contiguous().detach()
        target["lstm_c"] = self.cell.transpose(0, 1).contiguous().detach()

    def detach(self) -> "LSTMState":
        return LSTMState(self.hidden.detach(), self.cell.detach())
