from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_navigation


def mettagrid() -> MettaGridConfig:
    return make_navigation(num_agents=1)
