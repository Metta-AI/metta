import torch

import metta.agent.policy
import mettagrid.policy.policy_env_interface


class _DummyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(obs)


def _make_policy_env_info() -> mettagrid.policy.policy_env_interface.PolicyEnvInterface:
    import mettagrid.config

    return mettagrid.policy.policy_env_interface.PolicyEnvInterface.from_mg_cfg(mettagrid.config.MettaGridConfig())


def test_external_policy_wrapper_is_module() -> None:
    wrapper = metta.agent.policy.ExternalPolicyWrapper(_DummyPolicy(), _make_policy_env_info())

    # These nn.Module helpers should work without raising AttributeError
    wrapper.train()
    wrapper.eval()
    wrapper.to(torch.device("cpu"))

    assert isinstance(wrapper.policy, torch.nn.Module)
