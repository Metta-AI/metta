import pytest

from metta.mettagrid import dtype_actions, dtype_observations
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.mettagrid.mettagrid_env import MettaGridEnv
from tools.renderer import OpportunisticPolicy


@pytest.fixture
def env_with_agent_and_resource():
    env_cfg = EnvConfig.EmptyRoom(num_agents=1)
    env = MettaGridEnv(env_cfg, render_mode="human")
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
