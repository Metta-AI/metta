from omegaconf import OmegaConf

from metta.mettagrid.curriculum import single_task
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.room.random import Random
from metta.mettagrid.util.hydra import get_cfg


def test_env_map():
    cfg = get_cfg("benchmark")

    del cfg.game.map_builder
    cfg.game.num_agents = 1

    # Create a level with one agent
    level_builder = Random(width=3, height=4, objects=OmegaConf.create({}), agents=1, border_width=1)
    level = level_builder.build()

    curriculum = single_task("benchmark", cfg)
    env = MettaGridEnv(curriculum, render_mode="human", level=level)

    assert env.map_width == 3 + 2 * 1
    assert env.map_height == 4 + 2 * 1
