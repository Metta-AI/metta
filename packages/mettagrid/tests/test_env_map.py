import mettagrid.config.mettagrid_config
import mettagrid.simulator


def test_env_map():
    config = mettagrid.config.mettagrid_config.MettaGridConfig.EmptyRoom(
        width=3, height=4, num_agents=1, border_width=1
    )
    sim = mettagrid.simulator.Simulation(config)

    # The map dimensions should match the specified width/height
    assert sim.map_width == 3
    assert sim.map_height == 4
