import numpy as np

from mettagrid.simulator import Simulation, SimulatorEventHandler


class EarlyResetWrapper(SimulatorEventHandler):
    def __init__(self, simulator: Simulation):
        super().__init__(simulator)
        self._early_reset: int | None = int(np.random.randint(1, simulator.config.game.max_steps + 1))

    def on_step(self) -> None:
        if self._early_reset is not None and self._sim.state.current_step >= self._early_reset:
            self._sim.end_episode()
            self._early_reset = None
