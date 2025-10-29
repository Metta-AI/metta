from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from mettagrid.simulator.simulator import Simulation


@dataclass
class ObservationFeature:
    id: int
    name: str
    normalization: float


@dataclass
class ObservationToken:
    feature: ObservationFeature
    location: tuple[int, int]
    value: int

    def row(self) -> int:
        return self.location[1]

    def col(self) -> int:
        return self.location[0]


@dataclass
class AgentObservation:
    agent_id: int
    tokens: Sequence[ObservationToken]


@dataclass
class Action:
    name: str


class SimulatorEventHandler:
    """Handler for Simulator events."""

    def __init__(self):
        self._sim: Optional["Simulation"] = None

    def set_simulation(self, simulation: "Simulation") -> None:
        self._sim = simulation

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def on_close(self) -> None:
        pass
