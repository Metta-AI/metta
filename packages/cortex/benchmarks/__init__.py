"""Benchmark suite for Cortex kernels."""

# Import benchmark modules to populate the registry on module load.
# pylint: disable=unused-import
from . import axons as _axons  # noqa: F401
from . import conv1d as _conv1d  # noqa: F401
from . import linear_vs_axon as _linear_vs_axon  # noqa: F401
from . import lstm as _lstm  # noqa: F401
from . import mlstm as _mlstm  # noqa: F401
from . import rtu as _rtu  # noqa: F401
from . import slstm as _slstm  # noqa: F401
from .common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    get_registry,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkDefinition",
    "BenchmarkSettings",
    "get_registry",
]
