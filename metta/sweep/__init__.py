"""METTA Sweep System - Hyperparameter optimization and distributed execution.

This module provides:
- Hyperparameter sweep tools with Bayesian optimization
- Distributed job execution via worker pools
- Multiple dispatch strategies (local, cloud, queue-based)
- Parameter configuration and optimization
"""

# Core parameter configuration
from metta.sweep.core import (
    CategoricalParameterConfig,
    Distribution,
    ParameterConfig,
    SweepParameters,
    grid_search,
    make_sweep,
)

# Dispatchers for job execution
from metta.sweep.dispatchers import (
    LocalDispatcher,
    RemoteQueueDispatcher,
    SkypilotDispatcher,
)

# Protein optimizer configuration
from metta.sweep.protein_config import ProteinConfig, ProteinSettings

# High-level tools
from metta.sweep.tools import WorkerTool

__all__ = [
    # Core configuration
    "ParameterConfig",
    "CategoricalParameterConfig",
    "SweepParameters",
    "Distribution",
    "make_sweep",
    "grid_search",
    # Protein optimizer
    "ProteinConfig",
    "ProteinSettings",
    # Tools
    "WorkerTool",
    # Dispatchers
    "LocalDispatcher",
    "RemoteQueueDispatcher",
    "SkypilotDispatcher",
]
