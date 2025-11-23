import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

import mettagrid.builder.envs as eb
from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_artifact import save_policy_artifact_safetensors
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, policy_env_info):  # pragma: no cover - tests use provided agent
        return MockAgent()


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


class TestBasicSaveLoad:
    def test_load_from_uri_with_latest(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Test loading policy with :latest selector."""
        for epoch in [1, 7, 3]:
            save_policy_artifact_safetensors(
                checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v{epoch}.mpt",
                policy_architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}:latest"
        artifact = CheckpointManager.load_artifact_from_uri(latest_uri)

        assert artifact.state_dict is not None
        metadata = CheckpointManager.get_policy_metadata(latest_uri)
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 7

    def test_save_and_load_agent(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        expected_filename = "test_run:v5.mpt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        metadata = CheckpointManager.get_policy_metadata(agent_file.as_uri())
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 5

        artifact = CheckpointManager.load_artifact_from_uri(agent_file.as_uri())
        assert artifact.state_dict is not None

    def test_remote_prefix_upload(self, test_system_cfg, mock_agent, mock_policy_architecture):
        test_system_cfg.local_only = False
        test_system_cfg.remote_prefix = "s3://bucket/checkpoints"
        manager = CheckpointManager(run="test_run", system_cfg=test_system_cfg)

        expected_filename = "test_run:v3.mpt"
        local_path = manager.checkpoint_dir / expected_filename

        save_policy_artifact_safetensors(
            local_path,
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        # Simulate remote prefix using a local directory
        remote_dir = manager.checkpoint_dir / "remote"
        remote_dir.mkdir(parents=True, exist_ok=True)

        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy_spec = CheckpointManager.policy_spec_from_uri(f"file://{local_path}")
        policy = initialize_or_load_policy(env_info, policy_spec)
        remote_path = remote_dir / expected_filename
        remote_uri = policy.save_policy(remote_path, policy_architecture=mock_policy_architecture)

        assert remote_path.exists()
        # URI may percent-encode ':'; compare normalized parts
        assert Path(ParsedURI.parse(remote_uri).local_path) == remote_path
        assert remote_path.exists()

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        epochs = [1, 5, 10]

        for epoch in epochs:
            save_policy_artifact_safetensors(
                checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v{epoch}.mpt",
                policy_architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint.endswith(":v10.mpt")
        artifact = CheckpointManager.load_artifact_from_uri(latest_checkpoint)
        assert artifact.state_dict is not None

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        stopwatch_state = {"elapsed_time": 123.45}
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=5, agent_step=1000, stopwatch_state=stopwatch_state)

        loaded_trainer_state = checkpoint_manager.load_trainer_state()
        assert loaded_trainer_state is not None
        assert loaded_trainer_state["epoch"] == 5
        assert loaded_trainer_state["agent_step"] == 1000
        assert loaded_trainer_state["stopwatch_state"]["elapsed_time"] == 123.45
        assert loaded_trainer_state.get("loss_states", {}) == {}
        assert "optimizer_state" in loaded_trainer_state

    def test_checkpoint_existence(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None

        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v1.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

    def test_policy_spec_from_uri_initializes_checkpoint(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v1.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = CheckpointManager.policy_spec_from_uri(latest)
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)
        assert policy.agent_policy(0) is not None

    def test_checkpoint_policy_remains_callable(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v2.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = CheckpointManager.policy_spec_from_uri(latest)
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)

        assert callable(policy)

        obs_shape = env_info.observation_space.shape
        env_obs = torch.zeros((env_info.num_agents, *obs_shape), dtype=torch.uint8)
        td = TensorDict({"env_obs": env_obs}, batch_size=[env_info.num_agents])
        result = policy(td.clone())
        assert "actions" in result

    def test_policy_spec_display_name_preserved(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v4.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = CheckpointManager.policy_spec_from_uri(latest, display_name="friendly-name")
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)
        assert policy is not None
        assert spec.init_kwargs["display_name"] == "friendly-name"

    def test_checkpoint_policy_save_policy_round_trip(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = CheckpointManager.policy_spec_from_uri(latest)
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)

        save_path = checkpoint_manager.checkpoint_dir / "resaved.mpt"
        saved_uri = policy.save_policy(save_path, policy_architecture=mock_policy_architecture)

        assert save_path.exists()
        reload_spec = CheckpointManager.policy_spec_from_uri(saved_uri, device="cpu")
        reloaded = initialize_or_load_policy(env_info, reload_spec)
        assert reloaded is not None

    def test_policy_spec_from_uri(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact_safetensors(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v2.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

        spec = CheckpointManager.policy_spec_from_uri(latest, display_name="custom", device="cpu")
        assert isinstance(spec, PolicySpec)
        assert spec.class_path == mock_policy_architecture.class_path
        assert spec.init_kwargs is not None
        assert spec.init_kwargs["checkpoint_uri"] == CheckpointManager.normalize_uri(latest)
        assert spec.init_kwargs["display_name"] == "custom"
        assert spec.init_kwargs["device"] == "cpu"
        assert spec.init_kwargs.get("policy_architecture") is not None


class TestErrorHandling:
    def test_load_from_empty_directory(self, checkpoint_manager):
        result = checkpoint_manager.load_trainer_state()
        assert result is None

        checkpoints = checkpoint_manager.get_latest_checkpoint()
        assert checkpoints is None

    def test_invalid_run_name(self, test_system_cfg):
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=invalid_name, system_cfg=test_system_cfg)
