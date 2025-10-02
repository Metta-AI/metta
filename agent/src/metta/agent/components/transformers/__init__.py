"""Variant-specific transformer backbones."""

from __future__ import annotations

from typing import Dict

from . import gtrxl, sliding, trxl, trxl_nvidia  # noqa: F401 - ensures registration
from .spec import TransformerSpec

_SPECS: Dict[str, TransformerSpec] = {
    "gtrxl": gtrxl.SPEC,
    "trxl": trxl.SPEC,
    "trxl_nvidia": trxl_nvidia.SPEC,
    "sliding": sliding.SPEC,
}


def available_backbones() -> tuple[str, ...]:
    """Return the set of supported transformer backbone names."""

    return tuple(sorted(_SPECS.keys()))


def get_backbone_spec(name: str) -> TransformerSpec:
    """Fetch the registered spec for a backbone."""

    try:
        return _SPECS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown transformer backbone '{name}'") from exc


__all__ = ["TransformerSpec", "available_backbones", "get_backbone_spec"]
