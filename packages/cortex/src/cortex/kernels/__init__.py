"""Kernel implementations for Cortex cells.

This module contains various kernel implementations for cell computations,
including custom PyTorch kernels, Triton implementations, and optimized routines.
"""

from cortex.kernels.conv1d import TRITON_AVAILABLE as CONV1D_TRITON_AVAILABLE
from cortex.kernels.conv1d import causal_conv1d_pytorch
from cortex.kernels.mlstm import (
    TRITON_AVAILABLE,
    mlstm_chunkwise_simple,
    mlstm_chunkwise_triton,
    mlstm_recurrent_step_stabilized_simple,
)
from cortex.kernels.slstm import slstm_sequence_pytorch, slstm_sequence_triton

__all__ = [
    "TRITON_AVAILABLE",
    "CONV1D_TRITON_AVAILABLE",
    "causal_conv1d_pytorch",
    "mlstm_chunkwise_simple",
    "mlstm_chunkwise_triton",
    "mlstm_recurrent_step_stabilized_simple",
    "slstm_sequence_pytorch",
    "slstm_sequence_triton",
]

# Conditionally import and export Triton conv1d functions
if CONV1D_TRITON_AVAILABLE:
    from cortex.kernels.conv1d import causal_conv1d_triton  # noqa: F401

    __all__.append("causal_conv1d_triton")
