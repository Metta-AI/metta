"""Policy helper utilities, including LSTM state adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from cogames.policy.policy import PolicySpec

_POLICY_CLASS_SHORTHAND: dict[str, str] = {
    "random": "cogames.policy.random.RandomPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "token": "cogames.policy.token.TokenPolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
    "claude": "cogames.policy.claude.ClaudePolicy",
}


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "simple", "token", "lstm" or a full class path.

    Returns:
        Full class path to the policy.
    """
    return _POLICY_CLASS_SHORTHAND.get(policy, policy)


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    return {v: k for k, v in _POLICY_CLASS_SHORTHAND.items()}.get(policy)


def resolve_policy_data_path(policy_data_path: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint path if provided."""
    if policy_data_path is None:
        return None
    path = Path(policy_data_path)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    last_touched_checkpoint_file = max(
        (p for p in path.rglob("*.pt")),
        key=lambda target: target.stat().st_mtime,
        default=None,
    )
    if not last_touched_checkpoint_file:
        raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
    return str(last_touched_checkpoint_file)


def parse_policy_spec(spec: str) -> PolicySpec:
    """Parse a policy CLI option into its components."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = [part.strip() for part in raw.split(":")]
    if len(parts) > 3:
        raise ValueError("Policy specification must include at most two ':' separated values.")

    raw_class_path = parts[0]
    raw_policy_data = parts[1] if len(parts) > 1 else None
    raw_fraction = parts[2] if len(parts) > 2 else None

    if not raw_class_path:
        raise ValueError("Policy class path cannot be empty.")

    if not raw_fraction:
        fraction = 1.0
    else:
        try:
            fraction = float(raw_fraction)
        except ValueError as exc:
            raise ValueError(f"Invalid proportion value '{raw_fraction}'.") from exc

        if fraction <= 0:
            raise ValueError("Policy proportion must be a positive number.")

    resolved_class_path = resolve_policy_class_path(raw_class_path)
    resolved_policy_data = resolve_policy_data_path(raw_policy_data or None)

    return PolicySpec(
        policy_class_path=resolved_class_path,
        proportion=fraction,
        policy_data_path=resolved_policy_data,
    )


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


__all__ = [
    "PolicySpec",
    "resolve_policy_class_path",
    "resolve_policy_data_path",
    "parse_policy_spec",
    "LSTMState",
    "LSTMStateDict",
    "LSTMStateTuple",
]
