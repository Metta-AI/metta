from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.cortex import CortexTD
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from mettagrid.base_config import Config
from mettagrid.policy.mpt_artifact import MptArtifact, load_mpt, save_mpt
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class DummyActionComponentConfig(Config):
    name: str = "dummy"

    def make_component(self, env=None) -> nn.Module:
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

    def forward(self, td: TensorDict) -> TensorDict:
        td["logits"] = self.linear(td["env_obs"].float())
        return td

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_memory(self) -> None:
        return None


class ActionTestArchitecture(PolicyArchitecture):
    class_path: str = "tests.rl.test_policy_artifact.ActionTestPolicy"
    action_probs_config: DummyActionComponentConfig = Field(default_factory=DummyActionComponentConfig)


class ActionTestPolicy(Policy):
    def __init__(self, policy_env_info: PolicyEnvInterface | None, _: PolicyArchitecture | None = None):
        if policy_env_info is None:
            from mettagrid.config import MettaGridConfig

            policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)
        config = ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=4)
        self.components = nn.ModuleDict({"action_embedding": ActionEmbedding(config)})
        self._device = torch.device("cpu")

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        return td

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device):
        self._device = torch.device(device)
        self.components["action_embedding"].initialize_to_environment(policy_env_info, self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def reset_memory(self) -> None:
        return None


@pytest.fixture
def policy_env_info() -> PolicyEnvInterface:
    from mettagrid.config import MettaGridConfig

    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def test_mpt_round_trip(policy_env_info: PolicyEnvInterface, tmp_path: Path) -> None:
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    artifact_path = tmp_path / "artifact.mpt"
    save_mpt(artifact_path, architecture=architecture, state_dict=policy.state_dict())
    loaded = load_mpt(str(artifact_path))

    instantiated = loaded.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)
    assert instantiated.device.type == "cpu"


@pytest.mark.parametrize(
    "config, expectation",
    [
        (ViTDefaultConfig(), ViTDefaultConfig),
        (FastConfig(actor_hidden_dim=321), FastConfig),
    ],
)
def test_architecture_round_trip(config: PolicyArchitecture, expectation: type[PolicyArchitecture]) -> None:
    spec = config.to_spec()
    reconstructed = PolicyArchitecture.from_spec(spec)

    assert isinstance(reconstructed, expectation)
    assert reconstructed.model_dump() == config.model_dump()


def test_safetensors_save_with_fast_core(policy_env_info: PolicyEnvInterface, tmp_path: Path) -> None:
    architecture = FastConfig()
    policy = architecture.make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    artifact_path = tmp_path / "artifact.mpt"
    save_mpt(artifact_path, architecture=architecture, state_dict=policy.state_dict())

    reloaded = load_mpt(str(artifact_path)).instantiate(policy_env_info, torch.device("cpu"))

    assert hasattr(reloaded, "core")
    assert isinstance(reloaded.core, CortexTD)


def test_policy_artifact_reinitializes_environment_dependent_buffers(
    policy_env_info: PolicyEnvInterface,
) -> None:
    architecture = ActionTestArchitecture()
    policy = architecture.make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    artifact = MptArtifact(architecture=architecture, state_dict=policy.state_dict())
    artifact.state_dict["components.action_embedding.active_indices"] = torch.tensor([5, 6, 7], dtype=torch.long)

    reloaded = artifact.instantiate(policy_env_info, torch.device("cpu"))

    action_component = reloaded.components["action_embedding"]
    expected_indices = tuple(range(len(policy_env_info.action_names)))
    assert tuple(action_component.active_indices.tolist()) == expected_indices
    assert action_component.num_actions == len(expected_indices)
