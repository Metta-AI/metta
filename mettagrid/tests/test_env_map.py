from metta.mettagrid.gym_env import MettaGridGymEnv
from metta.mettagrid.mettagrid_config import EnvConfig


def test_env_map():
    env_config = EnvConfig.EmptyRoom(width=3, height=4, num_agents=1, border_width=1)
    env = MettaGridGymEnv(env_config=env_config, render_mode="human")

    # The map dimensions should match the specified width/height
    assert env.map_width == 3
    assert env.map_height == 4
