import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.hydra import get_cfg
from tools.renderer import OpportunisticPolicy, RandomPolicy, SimplePolicy, get_policy


@pytest.fixture
def tiny_env():
    cfg = get_cfg("benchmark")
    cfg.game.num_agents = 1
    cfg.game.max_steps = 5
    cfg.game.map_builder = OmegaConf.create(
        {
            "_target_": "metta.mettagrid.room.random.Random",
            "width": 5,
            "height": 5,
            "agents": 1,
            "border_width": 1,
            "objects": {},
        }
    )
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode="human")
    try:
        yield env
    finally:
        env.close()


def test_get_policy_selection_basic(tiny_env):
    dummy = OmegaConf.create({})
    assert isinstance(get_policy("random", tiny_env, dummy), RandomPolicy)
    assert isinstance(get_policy("simple", tiny_env, dummy), SimplePolicy)
    assert isinstance(get_policy("opportunistic", tiny_env, dummy), OpportunisticPolicy)
    # Unknown falls back to simple
    assert isinstance(get_policy("does_not_exist", tiny_env, dummy), SimplePolicy)
