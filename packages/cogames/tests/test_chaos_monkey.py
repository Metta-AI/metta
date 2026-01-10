import pytest

from cogames.policy.chaos_monkey import ChaosMonkeyPolicy
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def test_chaos_monkey_raises_at_fail_step() -> None:
    cfg = MettaGridConfig.EmptyRoom(num_agents=1, width=3, height=3, with_walls=False)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
    policy = ChaosMonkeyPolicy(policy_env_info, fail_step=1, fail_probability=1.0)
    agent = policy.agent_policy(0)
    obs = AgentObservation(agent_id=0, tokens=[])

    agent.step(obs)
    with pytest.raises(RuntimeError, match="Chaos monkey triggered at step 1"):
        agent.step(obs)
