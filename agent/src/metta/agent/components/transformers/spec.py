from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:  # pragma: no cover
    from metta.agent.components.transformer_core import TransformerBackboneConfig

BackboneBuilder = Callable[["TransformerBackboneConfig", Any | None], Any]


@dataclass(frozen=True)
class TransformerSpec:
    defaults: Dict[str, Any]
    policy_defaults: Dict[str, Any]
    builder: BackboneBuilder
