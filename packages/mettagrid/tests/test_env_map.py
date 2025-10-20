from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.mettagrid_env import MettaGridEnv


def test_env_map():
    config = MettaGridConfig.EmptyRoom(width=3, height=4, num_agents=1, border_width=1)
    env = MettaGridEnv(env_cfg=config, render_mode="none")

    # The map dimensions should match the specified width/height
    assert env.map_width == 3
    assert env.map_height == 4
