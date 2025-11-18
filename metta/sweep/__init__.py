"""METTA Sweep System - Hyperparameter optimization and distributed execution.

This module provides:
- Hyperparameter sweep tools with Bayesian optimization
- Distributed job execution via worker pools
- Multiple dispatch strategies (local, cloud, queue-based)
- Parameter configuration and optimization
"""

# Core parameter configuration
from metta.sweep.core import (
    ParameterConfig,
    CategoricalParameterConfig,
    SweepParameters,
    Distribution,
    make_sweep,
    grid_search,
)

# Protein optimizer configuration
from metta.sweep.protein_config import ProteinConfig, ProteinSettings

# Orchestration framework
from metta.sweep.experiment import SweepOrchestrator, Trial, TrialState, Observation
from metta.sweep.sweep import ProteinSweep

# Models and protocols
from metta.sweep.models import JobDefinition, JobTypes, JobStatus, RunInfo
from metta.sweep.protocols import Dispatcher, Store, Optimizer

# Stores
from metta.sweep.stores import WandbStore

# High-level tools
from metta.sweep.tools import SweepTool

# Dispatchers for job execution
from metta.sweep.dispatchers import (
    LocalDispatcher,
    SkypilotDispatcher,
)

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
    # Orchestration framework
    "SweepOrchestrator",
    "Trial",
    "TrialState",
    "Observation",
    "ProteinSweep",
    # Models
    "JobDefinition",
    "JobTypes",
    "JobStatus",
    "RunInfo",
    # Protocols
    "Dispatcher",
    "Store",
    "Optimizer",
    # Stores
    "WandbStore",
    # Tools
    "SweepTool",
    # Dispatchers
    "LocalDispatcher",
    "SkypilotDispatcher",
]
