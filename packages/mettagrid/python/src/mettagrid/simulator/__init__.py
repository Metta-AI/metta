from __future__ import annotations

import mettagrid.simulator.interface
import mettagrid.simulator.simulator

Simulation = mettagrid.simulator.simulator.Simulator

__all__ = [
    "Action",
    "AgentObservation",
    "ObservationToken",
    "SimulatorEventHandler",
    "Simulator",
    "Simulation",
]
