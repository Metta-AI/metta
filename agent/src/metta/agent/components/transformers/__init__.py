"""Variant-specific transformer backbones."""

from __future__ import annotations

from typing import Dict

from . import gtrxl, sliding, trxl, trxl_nvidia

_SPECS: Dict[str, Dict[str, object]] = {
    "gtrxl": gtrxl.SPEC,
    "trxl": trxl.SPEC,
    "trxl_nvidia": trxl_nvidia.SPEC,
    "sliding": sliding.SPEC,
}


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_SPECS.keys()))


def get_backbone_spec(name: str) -> Dict[str, object]:
    try:
        return _SPECS[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown transformer backbone '{name}'") from exc


__all__ = ["available_backbones", "get_backbone_spec"]
