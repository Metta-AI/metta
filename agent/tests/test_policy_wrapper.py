from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from metta.agent.policy import ExternalPolicyWrapper
from metta.rl.training import EnvironmentMetaData


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(obs)


def _make_env_metadata() -> EnvironmentMetaData:
    return EnvironmentMetaData(
        obs_width=4,
        obs_height=4,
        obs_features={},
        action_names=["noop"],
        num_agents=1,
        observation_space=SimpleNamespace(),
        action_space=SimpleNamespace(n=1, dtype=np.int32),
        feature_normalizations={0: 1.0},
    )


def test_external_policy_wrapper_is_module() -> None:
    wrapper = ExternalPolicyWrapper(_DummyPolicy(), _make_env_metadata())

    # These nn.Module helpers should work without raising AttributeError
    wrapper.train()
    wrapper.eval()
    wrapper.to(torch.device("cpu"))

    assert isinstance(wrapper.policy, nn.Module)
