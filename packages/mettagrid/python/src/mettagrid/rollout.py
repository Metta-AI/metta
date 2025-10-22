from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy import AgentPolicy
from mettagrid.simulator import Simulator


class Rollout:
    def __init__(
        self,
        config: MettaGridConfig,
        policies: list[AgentPolicy],
    ):
        self._config = config
        self._policies = policies
        self._simulator = Simulator(config)

    def rollout(self, num_steps: int) -> None:
        for _ in range(num_steps):
            actions = [policy.step(self._simulator.observations) for policy in self._policies]
            self._simulator.step(actions)
