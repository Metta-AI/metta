import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pydantic import Field

from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from mettagrid.config import Config


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, env_metadata):  # pragma: no cover - tests use provided agent
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
        checkpoint_manager.save_agent(mock_agent, epoch=1, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=7, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=3, policy_architecture=mock_policy_architecture)

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}:latest"
        artifact = CheckpointManager.load_artifact_from_uri(latest_uri)

        assert artifact.state_dict is not None
        metadata = CheckpointManager.get_policy_metadata(latest_uri)
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 7

    def test_save_and_load_agent(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        checkpoint_manager.save_agent(mock_agent, epoch=5, policy_architecture=mock_policy_architecture)

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        expected_filename = "test_run:v5.mpt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        metadata = CheckpointManager.get_policy_metadata(agent_file.as_uri())
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 5

        artifact = CheckpointManager.load_artifact_from_uri(agent_file.as_uri())
        assert artifact.state_dict is not None
        assert artifact.extra_files is None

    def test_save_agent_with_extra_files(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        payload = b"status: ready\n"
        checkpoint_manager.save_agent(
            mock_agent,
            epoch=2,
            policy_architecture=mock_policy_architecture,
            extra_files={"metadata/model_compatibility.yaml": payload},
        )

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        agent_file = checkpoint_dir / "test_run:v2.mpt"

        artifact = CheckpointManager.load_artifact_from_uri(agent_file.as_uri())
        assert artifact.extra_files is not None
        assert artifact.extra_files.get("metadata/model_compatibility.yaml") == payload

    def test_remote_prefix_upload(self, test_system_cfg, mock_agent, mock_policy_architecture):
        test_system_cfg.local_only = False
        test_system_cfg.remote_prefix = "s3://bucket/checkpoints"
        manager = CheckpointManager(run="test_run", system_cfg=test_system_cfg)

        expected_filename = "test_run:v3.mpt"
        expected_remote = f"s3://bucket/checkpoints/{expected_filename}"

        with patch("metta.rl.checkpoint_manager.write_file") as mock_write:
            remote_uri = manager.save_agent(mock_agent, epoch=3, policy_architecture=mock_policy_architecture)

        assert remote_uri == expected_remote
        mock_write.assert_called_once()
        remote_arg, local_arg = mock_write.call_args[0]
        assert remote_arg == expected_remote
        assert Path(local_arg).name == expected_filename

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        epochs = [1, 5, 10]

        for epoch in epochs:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, policy_architecture=mock_policy_architecture)

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint.endswith(":v10.mpt")
        artifact = CheckpointManager.load_artifact_from_uri(latest_checkpoint)
        assert artifact.state_dict is not None

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        checkpoint_manager.save_agent(mock_agent, epoch=5, policy_architecture=mock_policy_architecture)

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

        checkpoint_manager.save_agent(mock_agent, epoch=1, policy_architecture=mock_policy_architecture)
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None


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
