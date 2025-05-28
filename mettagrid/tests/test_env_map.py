from omegaconf import OmegaConf

import mettagrid.room.random
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.hydra import get_cfg


def test_env_map():
    cfg = get_cfg("benchmark")

    del cfg.game.map_builder
    cfg.game.num_agents = 0

    level_builder = mettagrid.room.random.Random(width=3, height=4, objects=OmegaConf.create({}), border_width=1)
    level = level_builder.build()

    env = MettaGridEnv(cfg, render_mode="human", level=level)

    assert env.map_width == 3 + 2 * 1
    assert env.map_height == 4 + 2 * 1
