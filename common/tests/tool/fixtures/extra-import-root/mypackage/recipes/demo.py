from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_arena


def mettagrid() -> MettaGridConfig:
    # Lightweight config generation; no environment execution
    return make_arena(num_agents=2)


def simulations() -> list[SimulationConfig]:
    env = mettagrid()
    return [SimulationConfig(suite="demo", name="basic", env=env)]
