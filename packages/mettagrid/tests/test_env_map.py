from mettagrid.config.mettagrid_config import MettaGridEnvConfig
from mettagrid.simulator import Simulation


def test_env_map():
    config = MettaGridEnvConfig.EmptyRoom(width=3, height=4, num_agents=1, border_width=1)
    sim = Simulation(config.game)

    # The map dimensions should match the specified width/height
    assert sim.map_width == 3
    assert sim.map_height == 4
