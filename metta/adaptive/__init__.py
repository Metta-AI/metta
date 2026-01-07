"""Simplified adaptive experiment orchestration for Metta."""

from typing import TYPE_CHECKING

# Core components
from .adaptive_config import AdaptiveConfig
from .adaptive_controller import AdaptiveController

# Implementations
from .dispatcher import LocalDispatcher, SkypilotDispatcher

# Models (with proper status logic copied from sweep models)
from .models import JobDefinition, JobStatus, RunInfo

# Protocols
from .protocols import Dispatcher, ExperimentScheduler, Store

if TYPE_CHECKING:
    from metta.adaptive.stores.wandb import WandbStore

__all__ = [
    # Core
    "AdaptiveConfig",
    "AdaptiveController",
    # Protocols
    "ExperimentScheduler",
    "Dispatcher",
    "Store",
    # Models
    "JobDefinition",
    "JobStatus",
    "RunInfo",
    # Implementations
    "LocalDispatcher",
    "SkypilotDispatcher",
    "WandbStore",
]


def __getattr__(name: str):
    """Dynamically import heavy submodules only when needed."""
    if name == "WandbStore":
        from metta.adaptive.stores.wandb import WandbStore

        return WandbStore
    raise AttributeError(name)
