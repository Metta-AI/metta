from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.cortex import CortexTD
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_bundle import write_checkpoint_bundle
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri


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
        # Use a large embedding table to accommodate the full action set (includes all vibes).
        config = ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=4, num_embeddings=196)
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


def _policy_env_info() -> PolicyEnvInterface:
    from mettagrid.config import MettaGridConfig

    return PolicyEnvInterface.from_mg_cfg(MettaGridConfig())


def test_save_and_load_weights_and_architecture(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = DummyPolicyArchitecture()
    policy = architecture.make_policy(policy_env_info)

    checkpoint_dir = tmp_path / "checkpoint"
    write_checkpoint_bundle(
        checkpoint_dir,
        policy_class_path=architecture.class_path,
        architecture_spec=architecture.to_spec(),
        state_dict=policy.state_dict(),
    )

    spec = policy_spec_from_uri(checkpoint_dir.as_uri())
    instantiated = initialize_or_load_policy(policy_env_info, spec)
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

    checkpoint_dir = tmp_path / "checkpoint"
    write_checkpoint_bundle(
        checkpoint_dir,
        policy_class_path=architecture.class_path,
        architecture_spec=architecture.to_spec(),
        state_dict=policy.state_dict(),
    )
    spec = policy_spec_from_uri(checkpoint_dir.as_uri())
    reloaded = initialize_or_load_policy(policy_env_info, spec)

    assert hasattr(reloaded, "core")
    assert isinstance(reloaded.core, CortexTD)


def test_checkpoint_bundle_reinitializes_environment_dependent_buffers(tmp_path: Path) -> None:
    policy_env_info = _policy_env_info()
    architecture = ActionTestArchitecture()
    # Save state before env init so env-derived buffers are empty in the checkpoint.
    policy = architecture.make_policy(policy_env_info)
    checkpoint_dir = tmp_path / "checkpoint"
    write_checkpoint_bundle(
        checkpoint_dir,
        policy_class_path=architecture.class_path,
        architecture_spec=architecture.to_spec(),
        state_dict=policy.state_dict(),
    )
    spec = policy_spec_from_uri(checkpoint_dir.as_uri())
    reloaded = initialize_or_load_policy(policy_env_info, spec)

    action_component = reloaded.components["action_embedding"]
    expected_indices = tuple(range(len(policy_env_info.action_names)))
    assert tuple(action_component.active_indices.tolist()) == expected_indices
    assert action_component.num_actions == len(expected_indices)
