import pytest
from omegaconf import OmegaConf

from metta.mettagrid import dtype_actions, dtype_observations
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.hydra import get_cfg
from tools.renderer import OpportunisticPolicy


@pytest.fixture
def env_with_agent_and_resource():
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
    obs, _ = env.reset()
    assert obs.dtype == dtype_observations
    try:
        yield env
    finally:
        env.close()


def test_opportunistic_roam_path(env_with_agent_and_resource):
    env = env_with_agent_and_resource
    policy = OpportunisticPolicy(env)
    obs, _ = env.reset()
    actions = policy.predict(obs)
    assert actions.dtype == dtype_actions
    assert actions.shape[0] == env.num_agents
