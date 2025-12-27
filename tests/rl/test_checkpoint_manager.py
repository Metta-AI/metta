"""Tests for CheckpointManager - the core checkpoint save/load flows used during training."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

import mettagrid.builder.envs as eb
from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTD
from metta.agent.mocks import MockAgent
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager, write_checkpoint_bundle, write_checkpoint_dir
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import (
    get_checkpoint_metadata,
    policy_spec_from_uri,
    resolve_uri,
)


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:
        return nn.Identity()


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, policy_env_info):
        return MockAgent()


class DummyActionComponentConfig(Config):
    name: str = "dummy"

    def make_component(self, env=None) -> nn.Module:
        return nn.Identity()


class DummyPolicyArchitecture(PolicyArchitecture):
    class_path: str = "tests.rl.test_checkpoint_manager.DummyPolicy"
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
    class_path: str = "tests.rl.test_checkpoint_manager.ActionTestPolicy"
    action_probs_config: DummyActionComponentConfig = Field(default_factory=DummyActionComponentConfig)


class ActionTestPolicy(Policy):
    def __init__(self, policy_env_info: PolicyEnvInterface | None, _: PolicyArchitecture | None = None):
        if policy_env_info is None:
            from mettagrid.config import MettaGridConfig

            policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)
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


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SystemConfig(data_dir=Path(tmpdir), local_only=True)


@pytest.fixture
def checkpoint_manager(test_system_cfg):
    return CheckpointManager(run="test_run", system_cfg=test_system_cfg)


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_policy_architecture():
    return MockAgentPolicyArchitecture()


class TestCheckpointManagerFlows:
    """Test the actual flows that happen during training and evaluation."""

    def test_get_latest_checkpoint_returns_highest_epoch(
        self, checkpoint_manager, mock_agent, mock_policy_architecture
    ):
        """During training resume, we need to find the latest checkpoint."""
        for epoch in [1, 5, 10]:
            write_checkpoint_dir(
                base_dir=checkpoint_manager.checkpoint_dir,
                run_name=checkpoint_manager.run_name,
                epoch=epoch,
                architecture_spec=mock_policy_architecture.to_spec(),
                state_dict=mock_agent.state_dict(),
            )

        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert ":v10" in latest or "%3Av10" in latest

    def test_trainer_state_save_and_restore(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Trainer state must be saved alongside policy for proper resume."""
        write_checkpoint_dir(
            base_dir=checkpoint_manager.checkpoint_dir,
            run_name=checkpoint_manager.run_name,
            epoch=5,
            architecture_spec=mock_policy_architecture.to_spec(),
            state_dict=mock_agent.state_dict(),
        )

        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(
            mock_optimizer, epoch=5, agent_step=1000, stopwatch_state={"elapsed_time": 123.45}
        )

        loaded = checkpoint_manager.load_trainer_state()
        assert loaded is not None
        assert loaded["epoch"] == 5
        assert loaded["agent_step"] == 1000
        assert loaded["stopwatch_state"]["elapsed_time"] == 123.45
        assert "optimizer" in loaded

    def test_resolve_latest_uri(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """The :latest suffix is used by eval tools to find the newest checkpoint."""
        for epoch in [1, 7, 3]:
            write_checkpoint_dir(
                base_dir=checkpoint_manager.checkpoint_dir,
                run_name=checkpoint_manager.run_name,
                epoch=epoch,
                architecture_spec=mock_policy_architecture.to_spec(),
                state_dict=mock_agent.state_dict(),
            )

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}:latest"
        metadata = get_checkpoint_metadata(resolve_uri(latest_uri).canonical)
        assert metadata.epoch == 7

    def test_checkpoint_bundle_loads_and_runs(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Checkpoint bundle must load and produce actions."""
        write_checkpoint_dir(
            base_dir=checkpoint_manager.checkpoint_dir,
            run_name=checkpoint_manager.run_name,
            epoch=1,
            architecture_spec=mock_policy_architecture.to_spec(),
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        spec = policy_spec_from_uri(latest)
        policy = initialize_or_load_policy(env_info, spec)

        obs_shape = env_info.observation_space.shape
        env_obs = torch.zeros((env_info.num_agents, *obs_shape), dtype=torch.uint8)
        td = TensorDict({"env_obs": env_obs}, batch_size=[env_info.num_agents])
        result = policy(td.clone())
        assert "actions" in result


class TestCheckpointManagerValidation:
    def test_empty_directory_returns_none(self, checkpoint_manager):
        assert checkpoint_manager.load_trainer_state() is None
        assert checkpoint_manager.get_latest_checkpoint() is None


class TestCheckpointBundles:
    def test_save_and_load_weights_and_architecture(self, tmp_path: Path) -> None:
        policy_env_info = _policy_env_info()
        architecture = DummyPolicyArchitecture()
        policy = architecture.make_policy(policy_env_info)

        checkpoint_dir = tmp_path / "checkpoint"
        write_checkpoint_bundle(
            checkpoint_dir,
            architecture_spec=architecture.to_spec(),
            state_dict=policy.state_dict(),
        )

        spec = policy_spec_from_uri(checkpoint_dir.as_uri())
        instantiated = initialize_or_load_policy(policy_env_info, spec)
        assert isinstance(instantiated, DummyPolicy)

    def test_architecture_round_trip_vit(self) -> None:
        config = ViTDefaultConfig()
        spec = config.to_spec()
        reconstructed = PolicyArchitecture.from_spec(spec)

        assert isinstance(reconstructed, ViTDefaultConfig)
        assert reconstructed.model_dump() == config.model_dump()

    def test_architecture_round_trip_fast_with_override(self) -> None:
        config = FastConfig(actor_hidden_dim=321)
        spec = config.to_spec()
        reconstructed = PolicyArchitecture.from_spec(spec)

        assert isinstance(reconstructed, FastConfig)
        assert reconstructed.actor_hidden_dim == 321
        assert reconstructed.critic_hidden_dim == config.critic_hidden_dim

    def test_architecture_from_spec_without_args(self) -> None:
        spec = "metta.agent.policies.vit.ViTDefaultConfig"
        architecture = PolicyArchitecture.from_spec(spec)
        assert isinstance(architecture, ViTDefaultConfig)

        canonical = architecture.to_spec()
        assert canonical.startswith("metta.agent.policies.vit.ViTDefaultConfig(")
        round_tripped = PolicyArchitecture.from_spec(canonical)
        assert round_tripped.model_dump() == architecture.model_dump()

    def test_architecture_from_spec_with_args_round_trip(self) -> None:
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

    def test_safetensors_save_with_fast_core(self, tmp_path: Path) -> None:
        from mettagrid.config import MettaGridConfig

        policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())

        architecture = FastConfig()
        policy = architecture.make_policy(policy_env_info)
        policy.initialize_to_environment(policy_env_info, torch.device("cpu"))

        checkpoint_dir = tmp_path / "checkpoint"
        write_checkpoint_bundle(
            checkpoint_dir,
            architecture_spec=architecture.to_spec(),
            state_dict=policy.state_dict(),
        )

        spec = policy_spec_from_uri(checkpoint_dir.as_uri())
        reloaded = initialize_or_load_policy(policy_env_info, spec)

        assert hasattr(reloaded, "core")
        assert isinstance(reloaded.core, CortexTD)

    def test_checkpoint_bundle_reinitializes_environment_dependent_buffers(self, tmp_path: Path) -> None:
        policy_env_info = _policy_env_info()
        architecture = ActionTestArchitecture()
        policy = architecture.make_policy(policy_env_info)

        checkpoint_dir = tmp_path / "checkpoint"
        write_checkpoint_bundle(
            checkpoint_dir,
            architecture_spec=architecture.to_spec(),
            state_dict=policy.state_dict(),
        )
        spec = policy_spec_from_uri(checkpoint_dir.as_uri())
        reloaded = initialize_or_load_policy(policy_env_info, spec)

        action_component = reloaded.components["action_embedding"]
        expected_indices = tuple(range(len(policy_env_info.action_names)))
        assert tuple(action_component.active_indices.tolist()) == expected_indices
        assert action_component.num_actions == len(expected_indices)
