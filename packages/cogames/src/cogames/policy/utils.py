"""Policy helper utilities, including LSTM state adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from cogames.policy.policy import PolicySpec

if TYPE_CHECKING:  # pragma: no cover - optional console for CLI
    from rich.console import Console


@dataclass
class ActionLayout:
    """Helper for mapping between flattened action logits and (verb, arg) pairs."""

    max_args: np.ndarray  # size: num_verbs, each entry is max arg value for that verb
    starts: np.ndarray  # size: num_verbs, starting index of each verb block

    @property
    def total_actions(self) -> int:
        return int(self.max_args.sum() + len(self.max_args))

    @classmethod
    def from_env(cls, env: object) -> "ActionLayout":
        max_args: Iterable[int] = env.max_action_args  # type: ignore[attr-defined]
        array = np.asarray(list(max_args), dtype=np.int64)
        counts = array + 1
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        return cls(max_args=array, starts=starts)

    def clamp_args_numpy(self, verb_indices: np.ndarray, arg_indices: np.ndarray) -> np.ndarray:
        verbs = verb_indices.astype(np.int64, copy=False)
        args = arg_indices.astype(np.int64, copy=False)
        max_allowed = self.max_args[verbs]
        return np.minimum(args, max_allowed)

    def decode_numpy(self, flat_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(flat_indices, dtype=np.int64)
        verbs = np.searchsorted(self.starts[1:], idx, side="right")
        args = idx - self.starts[verbs]
        args = self.clamp_args_numpy(verbs, args)
        return np.stack([verbs, args], axis=-1)

    def encode_torch(self, verb_arg: torch.Tensor) -> torch.Tensor:
        """Convert tensor of shape (..., 2) to flattened indices."""

        if verb_arg.numel() == 0:
            return torch.empty_like(verb_arg[..., 0])

        starts = torch.as_tensor(self.starts, device=verb_arg.device, dtype=torch.long)
        max_args = torch.as_tensor(self.max_args, device=verb_arg.device, dtype=torch.long)
        verbs = verb_arg[..., 0].long()
        args = torch.minimum(verb_arg[..., 1].long(), max_args[verbs])
        return starts[verbs] + args

    def decode_torch(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Convert flattened indices to (..., 2) tensor on same device."""

        if flat_indices.numel() == 0:
            shape = flat_indices.shape + (2,)
            return torch.empty(*shape, device=flat_indices.device, dtype=torch.long)

        starts = torch.as_tensor(self.starts, device=flat_indices.device, dtype=torch.long)
        counts = torch.as_tensor(self.max_args + 1, device=flat_indices.device, dtype=torch.long)

        # torch.bucketize expects boundaries; use cumulative starts of next block
        boundaries = torch.cumsum(counts, dim=0)[:-1]
        verbs = torch.bucketize(flat_indices.long(), boundaries, right=False)
        args = flat_indices.long() - starts[verbs]
        max_args = torch.as_tensor(self.max_args, device=flat_indices.device, dtype=torch.long)
        args = torch.minimum(args, max_args[verbs])
        return torch.stack([verbs, args], dim=-1)


_POLICY_CLASS_SHORTHAND: dict[str, str] = {
    "random": "cogames.policy.random.RandomPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
    "claude": "cogames.policy.claude.ClaudePolicy",
}


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "simple", "lstm" or a full class path.

    Returns:
        Full class path to the policy.
    """
    return _POLICY_CLASS_SHORTHAND.get(policy, policy)


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    return {v: k for k, v in _POLICY_CLASS_SHORTHAND.items()}.get(policy)


def resolve_policy_data_path(
    policy_data_path: Optional[str],
    *,
    policy_class_path: Optional[str] = None,
    game_name: Optional[str] = None,
    console: Optional["Console"] = None,
) -> Optional[str]:
    """Resolve a checkpoint path if provided.

    Only local filesystem lookups are supported. If the requested path (or latest
    ``*.pt`` file inside a directory) cannot be found, ``FileNotFoundError`` is raised.
    The ``policy_class_path``/``game_name``/``console`` parameters remain for
    backward compatibility with existing call sites but are unused.
    """

    _ = (policy_class_path, game_name, console)

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

    if path.exists():  # Non-pt extension but present
        return str(path)

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def parse_policy_spec(
    spec: str,
    *,
    console: Optional["Console"] = None,
    game_name: Optional[str] = None,
) -> PolicySpec:
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
    resolved_policy_data = resolve_policy_data_path(
        raw_policy_data or None,
        policy_class_path=resolved_class_path,
        game_name=game_name,
        console=console,
    )

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
    "ActionLayout",
    "PolicySpec",
    "resolve_policy_class_path",
    "get_policy_class_shorthand",
    "resolve_policy_data_path",
    "parse_policy_spec",
    "LSTMState",
    "LSTMStateDict",
    "LSTMStateTuple",
]
