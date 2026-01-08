from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from tribal_village_env.cogames.policy import TribalPolicyEnvInfo, TribalVillagePufferPolicy

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.loader import discover_and_register_policies, resolve_policy_class_path
from mettagrid.simulator.interface import AgentObservation, ObservationToken


def test_policy_env_info_names_and_actions():
    info = TribalPolicyEnvInfo(
        observation_space=spaces.Box(0, 255, (3, 3, 3), dtype=np.uint8),
        action_space=spaces.Discrete(4),
        num_agents=2,
    )

    assert info.action_names == ["action_0", "action_1", "action_2", "action_3"]
    actions = info.actions
    assert len(actions) == 4
    assert actions[0].name == "action_0"


def test_policy_short_name_registration():
    discover_and_register_policies("tribal_village_env.cogames")
    path = resolve_policy_class_path("tribal")
    assert path.endswith("TribalVillagePufferPolicy")


def test_policy_uses_agent_observation_tokens() -> None:
    obs_space = spaces.Box(0, 255, (2, 3), dtype=np.uint8)
    action_space = spaces.Discrete(2)
    info = TribalPolicyEnvInfo(observation_space=obs_space, action_space=action_space, num_agents=1)
    policy = TribalVillagePufferPolicy(policy_env_info=info)

    class _StubNet(torch.nn.Module):
        def __init__(self, num_actions: int):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self._num_actions = num_actions
            self.last_obs: torch.Tensor | None = None

        def forward_eval(self, observations: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, None]:
            self.last_obs = observations
            batch = observations.shape[0]
            logits = torch.zeros((batch, self._num_actions), device=observations.device, dtype=torch.float32)
            return logits, None

    stub_net = _StubNet(num_actions=action_space.n)
    policy._net = stub_net
    policy._device = next(stub_net.parameters()).device

    tokens = [
        ObservationToken(
            feature=ObservationFeatureSpec(id=0, name="f0", normalization=1.0),
            location=(0, 0),
            value=1,
            raw_token=(9, 8, 7),
        ),
        ObservationToken(
            feature=ObservationFeatureSpec(id=1, name="f1", normalization=1.0),
            location=(0, 1),
            value=2,
            raw_token=(6, 5, 4),
        ),
    ]
    obs = AgentObservation(agent_id=0, tokens=tokens)

    policy.step(obs)

    assert stub_net.last_obs is not None
    expected = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=torch.float32) * (1.0 / 255.0)
    assert torch.allclose(stub_net.last_obs.squeeze(0), expected)
