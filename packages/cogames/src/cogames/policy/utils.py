"""Policy helper utilities, including LSTM state adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch

from cogames.aws_storage import DownloadOutcome, maybe_download_checkpoint
from cogames.policy.policy import PolicySpec

if TYPE_CHECKING:
    from rich.console import Console


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path."""
    return {
        "random": "cogames.policy.random.RandomPolicy",
        "simple": "cogames.policy.simple.SimplePolicy",
        "lstm": "cogames.policy.lstm.LSTMPolicy",
        "claude": "cogames.policy.claude.ClaudePolicy",
    }.get(policy, policy)


def resolve_policy_data_path(
    policy_data_path: Optional[str],
    *,
    policy_class_path: Optional[str] = None,
    game_name: Optional[str] = None,
    console: Optional["Console"] = None,
) -> Optional[str]:
    """Resolve the checkpoint path, downloading from S3 when configured."""

    if policy_data_path is None:
        return None

    path = Path(policy_data_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        latest_checkpoint = max(
            (candidate for candidate in path.rglob("*.pt")),
            key=lambda candidate: candidate.stat().st_mtime,
            default=None,
        )
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
        return str(latest_checkpoint)

    if path.exists():
        return str(path)

    if console is not None and policy_class_path is not None and path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome: DownloadOutcome = maybe_download_checkpoint(
            policy_path=path,
            game_name=game_name,
            policy_class_path=policy_class_path,
            console=console,
        )
        if outcome.downloaded:
            return str(path)

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def parse_policy_spec(
    spec: str,
    *,
    console: Optional["Console"] = None,
    game_name: Optional[str] = None,
) -> PolicySpec:
    """Parse a CLI policy specification string."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = [part.strip() for part in raw.split(":")]
    if len(parts) > 3:
        raise ValueError("Policy specification must include at most two ':' separators.")

    raw_class_path = parts[0]
    raw_policy_data = parts[1] if len(parts) > 1 else None
    raw_proportion = parts[2] if len(parts) > 2 else None

    if not raw_class_path:
        raise ValueError("Policy class path cannot be empty.")

    if raw_proportion is None:
        proportion = 1.0
    else:
        try:
            proportion = float(raw_proportion)
        except ValueError as exc:  # pragma: no cover - user input error path
            raise ValueError(f"Invalid proportion value '{raw_proportion}'.") from exc
        if proportion <= 0:
            raise ValueError("Policy proportion must be a positive number.")

    resolved_class_path = resolve_policy_class_path(raw_class_path)
    resolved_policy_data = resolve_policy_data_path(
        raw_policy_data,
        policy_class_path=resolved_class_path,
        game_name=game_name,
        console=console,
    )

    return PolicySpec(
        policy_class_path=resolved_class_path,
        policy_data_path=resolved_policy_data,
        proportion=proportion,
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
