from omegaconf import OmegaConf

from metta.mettagrid import AutoResetEnv
from metta.mettagrid.config import EnvConfig, PyGameConfig
from metta.mettagrid.room.random import Random
from metta.mettagrid.util.hydra import get_cfg


def test_env_map():
    cfg = get_cfg("benchmark")

    del cfg.game.map_builder
    cfg.game.num_agents = 1

    # Create a level with one agent
    level_builder = Random(width=3, height=4, objects=OmegaConf.create({}), agents=1, border_width=1)
    level = level_builder.build()

    # Create EnvConfig directly
    game_config_dict = OmegaConf.to_container(cfg.game)
    game_config = PyGameConfig(**game_config_dict)
    env_config = EnvConfig(game=game_config, level_map=level)

    env = AutoResetEnv(env_config=env_config, render_mode="human")

    assert env.map_width == 3 + 2 * 1
    assert env.map_height == 4 + 2 * 1
