from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.policy_artifact import (
    PolicyArtifact,
    load_policy_artifact,
    policy_architecture_to_string,
    save_policy_artifact,
)
from mettagrid.base_config import Config
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class DummyActionComponentConfig(Config):
    name: str = "dummy"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class DummyPolicyArchitecture(PolicyArchitecture):
    class_path: str = "tests.rl.test_policy_artifact.DummyPolicy"
    action_probs_config: DummyActionComponentConfig = Field(default_factory=DummyActionComponentConfig)


class DummyPolicy(Policy):
    def __init__(self, policy_env_info: PolicyEnvInterface | None, _: PolicyArchitecture | None = None):
        if policy_env_info is None:
            from mettagrid.config import MettaGridConfig

            policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)
        self.linear = nn.Linear(1, 1)

    def forward(self, td: TensorDict) -> TensorDict:  # pragma: no cover - simple passthrough
        td["logits"] = self.linear(td["env_obs"].float())
        return td

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_memory(self) -> None:  # pragma: no cover - no-op for dummy policy
        return None


def _policy_env_info() -> PolicyEnvInterface:
    from mettagrid.config import MettaGridConfig

    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def test_embedded_policy_instantiates() -> None:
    policy_env_info = _policy_env_info()
    policy = DummyPolicy(policy_env_info)

    artifact = PolicyArtifact(policy=policy)
    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))

    assert instantiated is policy
    assert instantiated.device.type == "cpu"


def test_safetensors_save_and_load(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    artifact_path = tmp_path / "artifact.mpt"
    artifact = save_policy_artifact(
        artifact_path,
        policy_architecture=architecture,
        state_dict=policy.state_dict(),
    )

    assert artifact_path.exists()
    assert artifact.policy_architecture is architecture
    assert artifact.state_dict is not None

    loaded = load_policy_artifact(artifact_path)
    instantiated = loaded.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_pt_state_dict_load(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    pt_path = tmp_path / "checkpoint.pt"
    torch.save(policy.state_dict(), pt_path)

    artifact = load_policy_artifact(pt_path)
    assert artifact.state_dict is not None
    artifact.policy_architecture = architecture
    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_policy_architecture_round_trip() -> None:
    config = ViTDefaultConfig()
    spec = policy_architecture_to_string(config)
    round_tripped = policy_architecture_to_string(policy_architecture_to_string(config))  # deterministic formatting
    assert spec.startswith("metta.agent.policies.vit.ViTDefaultConfig")
    assert round_tripped.startswith("metta.agent.policies.vit.ViTDefaultConfig")
