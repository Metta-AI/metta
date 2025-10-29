from __future__ import annotations

import torch
from torch import nn

from metta.agent.policy import ExternalPolicyWrapper
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(obs)


def _make_policy_env_info() -> PolicyEnvInterface:
    from mettagrid.config import MettaGridConfig

    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def test_external_policy_wrapper_is_module() -> None:
    wrapper = ExternalPolicyWrapper(_DummyPolicy(), _make_policy_env_info())

    # These nn.Module helpers should work without raising AttributeError
    wrapper.train()
    wrapper.eval()
    wrapper.to(torch.device("cpu"))

    assert isinstance(wrapper.policy, nn.Module)
