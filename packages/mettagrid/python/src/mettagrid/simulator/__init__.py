"""MettaGrid Simulator - Core simulation interface and implementation.

This module provides the main simulation interface for MettaGrid environments.
"""

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.simulator.interface import (
    Action,
    AgentObservation,
    ObservationToken,
    SimulatorEventHandler,
)
from mettagrid.simulator.simulator import (
    BoundingBox,
    Simulation,
    SimulationAgent,
    Simulator,
)

__all__ = [
    # From interface
    "Action",
    "AgentObservation",
    "ObservationFeatureSpec",
    "ObservationToken",
    "SimulatorEventHandler",
    # From simulator
    "BoundingBox",
    "Simulation",
    "SimulationAgent",
    "Simulator",
]
