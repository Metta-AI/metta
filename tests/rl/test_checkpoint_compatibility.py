from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
from pydantic import Field

from metta.agent.components.component_config import ComponentConfig
from metta.agent.migration.checkpoint_compatibility import check_checkpoint_compatibility
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.policy_artifact import save_policy_artifact_safetensors
from mettagrid.config import Config


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, env_metadata):  # pragma: no cover - tests use provided agent
        return MockAgent()


def _env_metadata() -> SimpleNamespace:
    obs_feature = SimpleNamespace(id=0, normalization=1.0)
    return SimpleNamespace(
        obs_width=1,
        obs_height=1,
        obs_features={"grid": obs_feature},
        action_names=["noop"],
        num_agents=1,
        observation_space=None,
        action_space=None,
        feature_normalizations={},
    )


def _state_dict_with_incompatibilities(state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    modified = OrderedDict((k, v.clone()) for k, v in state.items())
    keys = list(modified.keys())

    if keys:
        modified.pop(keys[0], None)  # trigger missing key

    if len(keys) > 1:
        key = keys[1]
        tensor = modified[key]
        if tensor.ndim >= 1 and tensor.shape[0] > 1:
            modified[key] = tensor[: tensor.shape[0] // 2]
        else:
            modified[key] = tensor.to(torch.float64)

    return modified


def test_checkpoint_compatibility_success(tmp_path: Path):
    arch = MockAgentPolicyArchitecture()
    policy = MockAgent()

    checkpoint_path = tmp_path / "checkpoint_ok.mpt"
    save_policy_artifact_safetensors(
        checkpoint_path,
        policy_architecture=arch,
        state_dict=policy.state_dict(),
    )

    report = check_checkpoint_compatibility(
        checkpoint_path,
        policy_architecture=arch,
        env_metadata=_env_metadata(),
    )

    assert report.success
    assert not report.missing_keys
    assert not report.shape_mismatches
    assert not report.errors


def test_checkpoint_compatibility_reports_issues(tmp_path: Path):
    arch = MockAgentPolicyArchitecture()
    policy = MockAgent()
    incompatible_state = _state_dict_with_incompatibilities(policy.state_dict())

    checkpoint_path = tmp_path / "checkpoint_bad.mpt"
    save_policy_artifact_safetensors(
        checkpoint_path,
        policy_architecture=arch,
        state_dict=incompatible_state,
    )

    report = check_checkpoint_compatibility(
        checkpoint_path,
        policy_architecture=arch,
        env_metadata=_env_metadata(),
    )

    assert not report.success
    assert report.missing_keys
    assert report.shape_mismatches or report.dtype_mismatches
