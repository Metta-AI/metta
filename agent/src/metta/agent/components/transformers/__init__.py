"""Variant-specific transformer backbones."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Dict

from . import gtrxl, sliding, trxl, trxl_nvidia


@dataclass(frozen=True)
class TransformerBackboneSpec:
    """Immutable metadata bundle for a transformer backbone variant."""

    defaults: Mapping[str, Any]
    policy_defaults: Mapping[str, Any]
    builder: Callable[..., Any]


def _build_spec(module: ModuleType) -> TransformerBackboneSpec:
    """Construct a spec from a backbone module."""

    defaults_raw = getattr(module, "DEFAULTS", None)
    if defaults_raw is None:  # pragma: no cover
        raise ValueError(f"Backbone module '{module.__name__}' is missing DEFAULTS")
    if not isinstance(defaults_raw, Mapping):  # pragma: no cover
        raise TypeError(
            f"Backbone module '{module.__name__}' expected DEFAULTS to be a mapping, found {type(defaults_raw)!r}"
        )
    defaults = MappingProxyType(dict(defaults_raw))

    policy_defaults_raw = getattr(module, "POLICY_DEFAULTS", {})
    if not isinstance(policy_defaults_raw, Mapping):  # pragma: no cover
        raise TypeError(
            f"Backbone module '{module.__name__}' expected POLICY_DEFAULTS to be a mapping, "
            f"found {type(policy_defaults_raw)!r}"
        )

    builder = getattr(module, "build_backbone", None)
    if not callable(builder):  # pragma: no cover
        raise ValueError(f"Backbone module '{module.__name__}' must define a callable build_backbone")

    policy_defaults = MappingProxyType(dict(policy_defaults_raw))
    return TransformerBackboneSpec(defaults=defaults, policy_defaults=policy_defaults, builder=builder)


_BACKBONE_MODULES: Dict[str, ModuleType] = {
    "gtrxl": gtrxl,
    "trxl": trxl,
    "trxl_nvidia": trxl_nvidia,
    "sliding": sliding,
}

_SPECS: Dict[str, TransformerBackboneSpec] = {name: _build_spec(module) for name, module in _BACKBONE_MODULES.items()}


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_SPECS.keys()))


def get_backbone_spec(name: str) -> TransformerBackboneSpec:
    try:
        return _SPECS[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown transformer backbone '{name}'") from exc


__all__ = ["TransformerBackboneSpec", "available_backbones", "get_backbone_spec"]
