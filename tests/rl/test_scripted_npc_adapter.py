from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from metta.rl.npc.factory import create_scripted_policy_adapter, load_npc_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class _StubUnderlyingEnv:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents


class _StubCurriculumEnv:
    def __init__(self, num_agents: int) -> None:
        self._env = _StubUnderlyingEnv(num_agents)
        self.num_agents = num_agents


class _StubVectorEnv:
    def __init__(self, num_envs: int, num_agents: int) -> None:
        self.envs = [_StubCurriculumEnv(num_agents) for _ in range(num_envs)]
        self.driver_env = self.envs[0]._env


def _make_policy_env_info(num_agents: int, action_names: list[str]) -> PolicyEnvInterface:
    actions = SimpleNamespace(actions=lambda: [SimpleNamespace(name=name) for name in action_names])
    env_info = PolicyEnvInterface(
        obs_features=[],
        actions=actions,  # type: ignore[arg-type]
        num_agents=num_agents,
        observation_space=gym.spaces.Box(low=0, high=255, shape=(1, 3), dtype=np.uint8),
        action_space=gym.spaces.Discrete(len(action_names) or 1),
        obs_width=1,
        obs_height=1,
    )
    return env_info


def test_scripted_policy_adapter_emits_actions() -> None:
    vector_env = _StubVectorEnv(num_envs=1, num_agents=2)
    policy_env_info = _make_policy_env_info(num_agents=2, action_names=["noop", "east"])
    adapter = create_scripted_policy_adapter(
        class_path="tests.rl.fake_scripted_policy.FakeScriptedAgentPolicy",
        policy_kwargs={"multiplier": 10},
        policy_env_info=policy_env_info,
        vector_env=vector_env,
    )

    adapter.initialize_to_environment(policy_env_info, torch.device("cpu"))

    td = TensorDict(
        {
            "env_obs": torch.zeros(2, 1, 3, dtype=torch.uint8),
            "dones": torch.zeros(2, dtype=torch.float32),
        },
        batch_size=[2],
    )

    out = adapter.forward(td.clone())
    assert out["actions"].tolist() == [10, 11]

    td_reset = TensorDict(
        {
            "env_obs": torch.zeros(2, 1, 3, dtype=torch.uint8),
            "dones": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
        batch_size=[2],
    )
    out_reset = adapter.forward(td_reset)
    assert out_reset["actions"][0].item() > out["actions"][0].item()
    assert out_reset["actions"][1].item() == 11


def test_load_npc_policy_with_mock_uri() -> None:
    vector_env = _StubVectorEnv(num_envs=1, num_agents=1)
    policy_env_info = _make_policy_env_info(num_agents=1, action_names=["noop"])
    policy, descriptor = load_npc_policy(
        npc_policy_uri="mock://test_agent",
        npc_policy_class=None,
        npc_policy_kwargs={},
        policy_env_info=policy_env_info,
        device=torch.device("cpu"),
        vector_env=vector_env,
    )

    assert descriptor == "mock://test_agent"
    assert hasattr(policy, "forward")
