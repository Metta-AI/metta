from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.components.cortex import CortexTD
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.policy_artifact import (
    PolicyArtifact,
    load_policy_artifact,
    policy_architecture_from_string,
    policy_architecture_to_string,
    save_policy_artifact_safetensors,
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


def test_policy_only_artifact_instantiate() -> None:
    policy_env_info = _policy_env_info()
    policy = DummyPolicy(policy_env_info)

    artifact = PolicyArtifact(policy=policy)

    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert instantiated is policy
    assert instantiated.device.type == "cpu"


def test_save_and_load_weights_and_architecture(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    artifact_path = tmp_path / "artifact.mpt"
    artifact = save_policy_artifact_safetensors(
        artifact_path,
        policy_architecture=architecture,
        state_dict=policy.state_dict(),
    )

    assert artifact_path.exists()
    assert artifact.policy_architecture is architecture
    assert artifact.state_dict is not None

    loaded = load_policy_artifact(artifact_path)
    assert loaded.policy is None
    assert isinstance(loaded.policy_architecture, DummyPolicyArchitecture)
    assert loaded.state_dict is not None

    instantiated = loaded.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_policy_artifact_rejects_policy_and_weights() -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)
    state = policy.state_dict()

    with pytest.raises(ValueError):
        PolicyArtifact(policy_architecture=architecture, state_dict=state, policy=policy)


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


def test_safetensors_save_with_fast_core(tmp_path: Path) -> None:
    from mettagrid.config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())

    architecture = FastConfig()
    policy = architecture.make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

    artifact_path = tmp_path / "artifact.mpt"
    save_policy_artifact_safetensors(
        artifact_path,
        policy_architecture=architecture,
        state_dict=policy.state_dict(),
    )

    loaded = load_policy_artifact(artifact_path)
    reloaded = loaded.instantiate(policy_env_info, torch.device("cpu"))

    assert hasattr(reloaded, "core")
    assert isinstance(reloaded.core, CortexTD)


def test_load_pt_simple_state_dict(tmp_path: Path) -> None:
    """Test loading simple .pt file with state_dict."""
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    pt_path = tmp_path / "checkpoint.pt"
    torch.save(policy.state_dict(), pt_path)

    artifact = load_policy_artifact(pt_path)
    assert artifact.state_dict is not None
    assert artifact.policy_architecture is None
    assert artifact.policy is None

    # Can instantiate with architecture provided separately
    artifact.policy_architecture = architecture
    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_load_pt_with_ddp_normalization(tmp_path: Path) -> None:
    """Test that .pt files with DDP-style keys are normalized."""
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    # Create DDP-style state dict with 'module.' prefix
    ddp_style = {f"module.{k}": v.clone() for k, v in policy.state_dict().items()}
    pt_path = tmp_path / "ddp_checkpoint.pt"
    torch.save(ddp_style, pt_path)

    artifact = load_policy_artifact(pt_path)
    assert artifact.state_dict is not None

    # Keys should be normalized (module. prefix stripped)
    for key in artifact.state_dict.keys():
        assert not key.startswith("module.")

    artifact.policy_architecture = architecture
    instantiated = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert isinstance(instantiated, DummyPolicy)


def test_load_mpt_requires_architecture(tmp_path: Path) -> None:
    """Test that .mpt files must contain architecture."""
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    mpt_path = tmp_path / "checkpoint.mpt"
    save_policy_artifact_safetensors(
        mpt_path,
        policy_architecture=architecture,
        state_dict=policy.state_dict(),
    )

    artifact = load_policy_artifact(mpt_path)
    assert artifact.policy_architecture is not None
    assert artifact.state_dict is not None
    assert isinstance(artifact.policy_architecture, DummyPolicyArchitecture)


def test_load_mpt_with_embedded_policy_pt(tmp_path: Path) -> None:
    """Legacy .mpt archives with policy.pt should still load."""
    policy_env_info = _policy_env_info()
    policy = DummyPolicy(policy_env_info)
    with torch.no_grad():
        policy.linear.weight.fill_(2.0)
        policy.linear.bias.fill_(1.0)

    buffer = io.BytesIO()
    torch.save(policy, buffer)
    buffer.seek(0)

    mpt_path = tmp_path / "legacy.mpt"
    with zipfile.ZipFile(mpt_path, mode="w") as archive:
        archive.writestr("policy.pt", buffer.getvalue())

    artifact = load_policy_artifact(mpt_path)
    assert artifact.policy is not None

    reloaded = artifact.instantiate(policy_env_info, torch.device("cpu"))
    assert torch.allclose(reloaded.linear.weight, policy.linear.weight)
    assert torch.allclose(reloaded.linear.bias, policy.linear.bias)


def test_load_invalid_extension_raises(tmp_path: Path) -> None:
    """Test that unsupported extensions raise clear errors."""
    invalid_path = tmp_path / "checkpoint.pkl"
    invalid_path.touch()

    with pytest.raises(ValueError, match="Unsupported checkpoint extension.*Expected .mpt or .pt"):
        load_policy_artifact(invalid_path)


def test_load_nonexistent_file_raises(tmp_path: Path) -> None:
    """Test that missing files raise clear errors."""
    missing_path = tmp_path / "missing.mpt"

    with pytest.raises(FileNotFoundError, match="Policy artifact not found"):
        load_policy_artifact(missing_path)


def test_pt_with_non_state_dict_raises(tmp_path: Path) -> None:
    """Test that .pt files must contain state_dict mappings."""
    pt_path = tmp_path / "invalid.pt"
    torch.save("not a state dict", pt_path)

    with pytest.raises(TypeError, match=".pt file must contain a state_dict mapping"):
        load_policy_artifact(pt_path)
