"""Variant-specific transformer backbones and registry."""

from . import gtrxl, sliding, trxl, trxl_nvidia  # noqa: F401
from .registry import available_backbones, get_backbone_entry, register_backbone

__all__ = [
    "available_backbones",
    "get_backbone_entry",
    "register_backbone",
]
