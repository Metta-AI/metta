import mettagrid
import mettagrid.builder.envs


def mettagrid() -> mettagrid.MettaGridConfig:
    return mettagrid.builder.envs.make_navigation(num_agents=1)
