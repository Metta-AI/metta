"""Static exports for transformer backbone variants."""

from __future__ import annotations

from typing import Dict, Type

from .gtrxl import GTrXLConfig
from .sliding import SlidingTransformerBackboneConfig
from .trxl import TRXLConfig
from .trxl_nvidia import TRXLNvidiaConfig

TransformerBackboneConfigType = GTrXLConfig | TRXLConfig | TRXLNvidiaConfig | SlidingTransformerBackboneConfig

BACKBONE_CONFIGS: Dict[str, Type] = {
    "gtrxl": GTrXLConfig,
    "trxl": TRXLConfig,
    "trxl_nvidia": TRXLNvidiaConfig,
    "sliding": SlidingTransformerBackboneConfig,
}


__all__ = [
    "GTrXLConfig",
    "TRXLConfig",
    "TRXLNvidiaConfig",
    "SlidingTransformerBackboneConfig",
    "TransformerBackboneConfigType",
    "BACKBONE_CONFIGS",
]
