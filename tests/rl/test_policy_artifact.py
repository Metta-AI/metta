from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.policy_artifact import load_policy_artifact, save_policy_artifact
from metta.rl.training import EnvironmentMetaData
from mettagrid.config import Config


class DummyActionComponentConfig(Config):
    name: str = "dummy"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class DummyPolicyArchitecture(PolicyArchitecture):
    class_path: str = "tests.rl.test_policy_artifact.DummyPolicy"
    action_probs_config: DummyActionComponentConfig = Field(default_factory=DummyActionComponentConfig)


class DummyPolicy(Policy):
    def __init__(self, env_metadata: EnvironmentMetaData | None, _: PolicyArchitecture | None = None):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, td: TensorDict) -> TensorDict:  # pragma: no cover - simple passthrough
        td["logits"] = self.linear(td["env_obs"].float())
        return td

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_memory(self) -> None:  # pragma: no cover - no-op for dummy policy
        return None


def _env_metadata() -> EnvironmentMetaData:
    return EnvironmentMetaData(
        obs_width=1,
        obs_height=1,
        obs_features={},
        action_names=[],
        max_action_args=[],
        num_agents=1,
        observation_space=None,
        action_space=None,
        feature_normalizations={},
    )


def test_save_and_load_weights_and_architecture(tmp_path: Path) -> None:
    env_metadata = _env_metadata()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(env_metadata)

    artifact_path = tmp_path / "artifact.zip"
    artifact = save_policy_artifact(
        artifact_path,
        policy=policy,
        policy_architecture=architecture,
        include_policy=False,
    )

    assert artifact_path.exists()
    assert artifact.policy_architecture is architecture
    assert artifact.state_dict is not None

    loaded = load_policy_artifact(artifact_path)
    assert loaded.policy is None
    assert isinstance(loaded.policy_architecture, DummyPolicyArchitecture)
    assert loaded.state_dict is not None

    instantiated = loaded.instantiate(env_metadata, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_save_and_load_policy_only(tmp_path: Path) -> None:
    env_metadata = _env_metadata()
    policy = DummyPolicy(env_metadata)

    artifact_path = tmp_path / "policy_only.zip"
    artifact = save_policy_artifact(
        artifact_path,
        policy=policy,
        include_policy=True,
    )

    assert artifact.policy is policy
    assert artifact_path.exists()

    loaded = load_policy_artifact(artifact_path)
    assert loaded.policy_architecture is None
    assert loaded.state_dict is None
    assert isinstance(loaded.policy, DummyPolicy)


def test_save_and_load_all_components(tmp_path: Path) -> None:
    env_metadata = _env_metadata()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(env_metadata)

    artifact_path = tmp_path / "full_bundle.zip"
    artifact = save_policy_artifact(
        artifact_path,
        policy=policy,
        policy_architecture=architecture,
        include_policy=True,
    )

    assert artifact_path.exists()
    assert artifact.policy is policy
    assert artifact.state_dict is not None
    assert artifact.policy_architecture is architecture

    loaded = load_policy_artifact(artifact_path)
    assert isinstance(loaded.policy_architecture, DummyPolicyArchitecture)
    assert loaded.state_dict is not None
    assert isinstance(loaded.policy, DummyPolicy)

    instantiated = loaded.instantiate(env_metadata, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)
