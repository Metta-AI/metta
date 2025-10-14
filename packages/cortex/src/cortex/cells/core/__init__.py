"""Core Axons-based primitives.

This subpackage hosts the AxonCell (streaming RTU cell) and AxonLayer, a
stateful, linear-like module that wraps AxonCell with automatic state
management.
"""

from .axon_cell import AxonCell
from .axon_layer import AxonLayer

__all__ = ["AxonCell", "AxonLayer"]
