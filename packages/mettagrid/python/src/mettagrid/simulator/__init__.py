"""MettaGrid Simulator - Core simulation interface and implementation.

This module provides the main simulation interface for MettaGrid environments.
"""

from typing import TYPE_CHECKING

from mettagrid.simulator.interface import (
    Action,
    AgentObservation,
    ObservationToken,
    SimulatorEventHandler,
)
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.simulator import (
    BoundingBox,
    Simulation,
    SimulationAgent,
    Simulator,
)

if TYPE_CHECKING:
    from mettagrid.config.id_map import ObservationFeatureSpec as _ObservationFeatureSpec  # noqa: F401

__all__ = [
    # From interface
    "Action",
    "AgentObservation",
    "ObservationFeatureSpec",
    "ObservationToken",
    "SimulatorEventHandler",
    # From replay_log_writer
    "ReplayLogWriter",
    # From simulator
    "BoundingBox",
    "Simulation",
    "SimulationAgent",
    "Simulator",
]


def __getattr__(name: str):
    if name == "ObservationFeatureSpec":
        from mettagrid.config.id_map import ObservationFeatureSpec as _ObservationFeatureSpec

        globals()[name] = _ObservationFeatureSpec
        return _ObservationFeatureSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
