from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    import mettagrid.config.id_map

    ObservationFeatureSpec = mettagrid.config.id_map.ObservationFeatureSpec
    import mettagrid.simulator.simulator

    Simulator = mettagrid.simulator.simulator.Simulator
else:
    ObservationFeatureSpec = typing.Any
    Simulator = typing.Any


@dataclasses.dataclass
class ObservationToken:
    feature: mettagrid.config.id_map.ObservationFeatureSpec
    location: tuple[int, int]
    value: int

    def row(self) -> int:
        return self.location[1]

    def col(self) -> int:
        return self.location[0]


@dataclasses.dataclass
class AgentObservation:
    agent_id: int
    tokens: typing.Sequence[ObservationToken]


@dataclasses.dataclass
class Action:
    name: str


class SimulatorEventHandler:
    """Handler for Simulator events."""

    def __init__(self):
        self._sim: typing.Optional[mettagrid.simulator.simulator.Simulator] = None

    def set_simulation(self, simulation: mettagrid.simulator.simulator.Simulator) -> None:
        self._sim = simulation

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def on_close(self) -> None:
        pass
