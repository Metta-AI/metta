from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

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


def _policy_env_info() -> PolicyEnvInterface:
    from mettagrid.config import MettaGridConfig

    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def test_artifact_instantiate() -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    artifact = MptArtifact(architecture=architecture, state_dict=policy.state_dict())

    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)
    assert instantiated.device.type == "cpu"


def test_save_and_load_weights_and_architecture(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    artifact_path = tmp_path / "artifact.mpt"
    save_mpt(artifact_path, architecture=architecture, state_dict=policy.state_dict())

    assert artifact_path.exists()

    loaded = load_mpt(str(artifact_path))
    assert isinstance(loaded.architecture, DummyPolicyArchitecture)
    assert loaded.state_dict is not None

    instantiated = loaded.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_architecture_round_trip_vit() -> None:
    config = ViTDefaultConfig()
    spec = config.to_spec()
    reconstructed = PolicyArchitecture.from_spec(spec)

    assert isinstance(reconstructed, ViTDefaultConfig)
    assert reconstructed.model_dump() == config.model_dump()


def test_architecture_round_trip_fast_with_override() -> None:
    config = FastConfig(actor_hidden_dim=321)
    spec = config.to_spec()
    reconstructed = PolicyArchitecture.from_spec(spec)

    assert isinstance(reconstructed, FastConfig)
    assert reconstructed.model_dump() == config.model_dump()


def test_architecture_from_spec_without_args() -> None:
    spec = "metta.agent.policies.vit.ViTDefaultConfig"
    architecture = PolicyArchitecture.from_spec(spec)
    assert isinstance(architecture, ViTDefaultConfig)

    canonical = architecture.to_spec()
    assert canonical.startswith("metta.agent.policies.vit.ViTDefaultConfig(")
    round_tripped = PolicyArchitecture.from_spec(canonical)
    assert round_tripped.model_dump() == architecture.model_dump()


def test_architecture_from_spec_with_args_round_trip() -> None:
    spec = "metta.agent.policies.fast.FastConfig(actor_hidden_dim=2048, critic_hidden_dim=4096)"
    architecture = PolicyArchitecture.from_spec(spec)

    assert isinstance(architecture, FastConfig)
    assert architecture.actor_hidden_dim == 2048
    assert architecture.critic_hidden_dim == 4096

    canonical = architecture.to_spec()
    assert "actor_hidden_dim=2048" in canonical
    assert "critic_hidden_dim=4096" in canonical
    round_tripped = PolicyArchitecture.from_spec(canonical)
    assert round_tripped.model_dump() == architecture.model_dump()


def test_safetensors_save_with_fast_core(tmp_path: Path) -> None:
    from mettagrid.config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())

    architecture = FastConfig()
    policy = architecture.make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    artifact_path = tmp_path / "artifact.mpt"
    save_mpt(artifact_path, architecture=architecture, state_dict=policy.state_dict())

    loaded = load_mpt(str(artifact_path))
    reloaded = loaded.instantiate(policy_env_info, torch.device("cpu"))

    assert hasattr(reloaded, "core")
    assert isinstance(reloaded.core, CortexTD)
