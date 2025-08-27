from metta.mettagrid.config.envs import make_arena
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool


def play(
    map: str = "/Users/jacke/metta/mettagrid/configs/basic_plant_map.txt",
    num_agents: int = 1,
) -> PlayTool:
    env = make_arena(num_agents=num_agents)
    env.game.map_builder = AsciiMapBuilder.Config.from_uri(map)
    return PlayTool(sim=SimulationConfig(name="basic_plant", env=env))
