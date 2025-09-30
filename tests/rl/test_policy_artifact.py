from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.policies.fast import FastConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.policy_artifact import (
    PolicyArtifact,
    load_policy_artifact,
    policy_architecture_from_string,
    policy_architecture_to_string,
    save_policy_artifact,
)
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


def test_policy_only_artifact_instantiate() -> None:
    env_metadata = _env_metadata()
    policy = DummyPolicy(env_metadata)

    artifact = PolicyArtifact(policy=policy)

    instantiated = artifact.instantiate(env_metadata, torch.device("cpu"))
    assert instantiated is policy
    assert instantiated.device.type == "cpu"


def test_save_and_load_weights_and_architecture(tmp_path: Path) -> None:
    env_metadata = _env_metadata()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(env_metadata)

    artifact_path = tmp_path / "artifact.zip"
    artifact = save_policy_artifact(
        artifact_path,
        policy=policy,
        policy_architecture=architecture,
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


def test_policy_artifact_rejects_policy_and_weights() -> None:
    env_metadata = _env_metadata()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(env_metadata)
    state = policy.state_dict()

    with pytest.raises(ValueError):
        PolicyArtifact(policy_architecture=architecture, state_dict=state, policy=policy)


def test_save_policy_artifact_rejects_include_policy_with_weights(tmp_path: Path) -> None:
    env_metadata = _env_metadata()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(env_metadata)

    with pytest.raises(ValueError):
        save_policy_artifact(
            tmp_path / "invalid.mpt",
            policy=policy,
            policy_architecture=architecture,
            include_policy=True,
        )


def test_policy_architecture_round_trip_vit() -> None:
    config = ViTDefaultConfig()
    spec = policy_architecture_to_string(config)
    reconstructed = policy_architecture_from_string(spec)

    assert isinstance(reconstructed, ViTDefaultConfig)
    assert reconstructed.model_dump() == config.model_dump()


def test_policy_architecture_round_trip_fast_with_override() -> None:
    config = FastConfig(actor_hidden_dim=321)
    spec = policy_architecture_to_string(config)
    reconstructed = policy_architecture_from_string(spec)

    assert isinstance(reconstructed, FastConfig)
    assert reconstructed.model_dump() == config.model_dump()


def test_policy_architecture_from_string_without_args() -> None:
    spec = "metta.agent.policies.vit.ViTDefaultConfig"
    architecture = policy_architecture_from_string(spec)
    assert isinstance(architecture, ViTDefaultConfig)

    canonical = policy_architecture_to_string(architecture)
    assert canonical.startswith("metta.agent.policies.vit.ViTDefaultConfig(")
    # Canonical string should parse back to the same config
    round_tripped = policy_architecture_from_string(canonical)
    assert round_tripped.model_dump() == architecture.model_dump()


def test_policy_architecture_from_string_with_args_round_trip() -> None:
    spec = "metta.agent.policies.fast.FastConfig(actor_hidden_dim=2048, critic_hidden_dim=4096)"
    architecture = policy_architecture_from_string(spec)

    assert isinstance(architecture, FastConfig)
    assert architecture.actor_hidden_dim == 2048
    assert architecture.critic_hidden_dim == 4096

    canonical = policy_architecture_to_string(architecture)
    assert "actor_hidden_dim=2048" in canonical
    assert "critic_hidden_dim=4096" in canonical
    # Canonical string should parse back to the same config
    round_tripped = policy_architecture_from_string(canonical)
    assert round_tripped.model_dump() == architecture.model_dump()
