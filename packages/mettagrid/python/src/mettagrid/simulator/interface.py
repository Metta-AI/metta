from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mettagrid.config.id_map import ObservationFeatureSpec

if TYPE_CHECKING:
    from mettagrid.simulator.simulator import Simulation


@dataclass
class ObservationToken:
    feature: ObservationFeatureSpec
    location: tuple[int, int]
    value: int

    raw_token: tuple[int, int, int]

    def row(self) -> int:
        return self.location[1]

    def col(self) -> int:
        return self.location[0]


@dataclass
class AgentObservation:
    agent_id: int
    tokens: Sequence[ObservationToken]


class SimulatorEventHandler:
    """Handler for Simulator events."""

    def __init__(self):
        self._sim: Simulation

    def set_simulation(self, simulation: Simulation) -> None:
        self._sim = simulation

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def on_close(self) -> None:
        pass
