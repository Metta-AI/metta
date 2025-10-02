from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict


@dataclass
class TransformerBackboneEntry:
    defaults: Dict[str, Any]
    policy_defaults: Dict[str, Any]
    builder: Callable[["TransformerBackboneConfig", Any | None], Any]


_REGISTRY: Dict[str, TransformerBackboneEntry] = {}


def register_backbone(
    name: str,
    defaults: Dict[str, Any],
    policy_defaults: Dict[str, Any],
    builder: Callable[["TransformerBackboneConfig", Any | None], Any],
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Transformer backbone '{name}' already registered")
    _REGISTRY[name] = TransformerBackboneEntry(defaults, policy_defaults, builder)


def get_backbone_entry(name: str) -> TransformerBackboneEntry:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown transformer backbone '{name}'") from exc


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))

if TYPE_CHECKING:  # pragma: no cover
    from metta.agent.components.transformer_core import TransformerBackboneConfig
