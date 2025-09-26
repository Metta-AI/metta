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

    def make_policy(self, env_metadata):  # pragma: no cover - tests use pre-built policy
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
    def test_latest_selector_file_uri(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Test :latest selector for file:// URIs."""
        # Save multiple checkpoints
        checkpoint_manager.save_agent(mock_agent, epoch=1, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=5, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=3, policy_architecture=mock_policy_architecture)

        # Test :latest resolution
        from metta.rl.checkpoint_manager import key_and_version

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.mpt"
        run_name, epoch = key_and_version(latest_uri)

        assert run_name == "test_run"
        assert epoch == 5  # Should resolve to highest epoch

    def test_load_from_uri_with_latest(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Test loading policy with :latest selector."""
        # Save multiple checkpoints
        checkpoint_manager.save_agent(mock_agent, epoch=1, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=7, policy_architecture=mock_policy_architecture)
        checkpoint_manager.save_agent(mock_agent, epoch=3, policy_architecture=mock_policy_architecture)

        # Load using :latest selector
        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.mpt"
        artifact = CheckpointManager.load_from_uri(latest_uri)

        assert artifact.policy_architecture is not None and isinstance(
            artifact.policy_architecture, MockAgentPolicyArchitecture
        )
        # Verify it loaded the correct checkpoint by checking metadata
        metadata = CheckpointManager.get_policy_metadata(latest_uri)
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 7  # Should be the highest epoch

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

        # Test checkpoint selection
        latest_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1)
        assert len(latest_checkpoints) == 1
        assert latest_checkpoints[0].endswith("test_run:v10.mpt")

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        # Save agent checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=5, policy_architecture=mock_policy_architecture)

        # Create and save trainer state
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        stopwatch_state = {"elapsed_time": 123.45}
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=5, agent_step=1000, stopwatch_state=stopwatch_state)

        # Load trainer state
        loaded_trainer_state = checkpoint_manager.load_trainer_state()
        assert loaded_trainer_state is not None
        assert loaded_trainer_state["epoch"] == 5
        assert loaded_trainer_state["agent_step"] == 1000
        assert loaded_trainer_state["stopwatch_state"]["elapsed_time"] == 123.45
        assert loaded_trainer_state.get("loss_states", {}) == {}
        assert "optimizer_state" in loaded_trainer_state


class TestCleanup:
    def test_cleanup_old_checkpoints(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        # Save 10 checkpoints
        for epoch in range(1, 11):
            checkpoint_manager.save_agent(
                mock_agent,
                epoch=epoch,
                policy_architecture=mock_policy_architecture,
            )

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        checkpoint_files = [p for p in checkpoint_dir.glob("*.mpt") if ":v" in p.stem]
        assert len(checkpoint_files) == 10

        # Clean up, keeping only 5
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
        assert deleted_count == 5

        # Verify only 5 remain (latest ones: epochs 6-10)
        remaining_files = [p for p in checkpoint_dir.glob("*.mpt") if ":v" in p.stem]
        assert len(remaining_files) == 5

        remaining_epochs = sorted(int(f.stem.split(":v")[1]) for f in remaining_files)
        assert remaining_epochs == [6, 7, 8, 9, 10]

    def test_cleanup_with_trainer_state(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        # Save checkpoint and trainer state
        checkpoint_manager.save_agent(
            mock_agent,
            epoch=1,
            policy_architecture=mock_policy_architecture,
        )
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=1, agent_step=1000)

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        assert (checkpoint_dir / "test_run:v1.mpt").exists()
        assert (checkpoint_dir / "trainer_state.pt").exists()

        # Cleanup should remove both
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=0)
        assert deleted_count == 1
        assert not (checkpoint_dir / "test_run:v1.mpt").exists()
        assert not (checkpoint_dir / "trainer_state.pt").exists()


class TestErrorHandling:
    def test_trainer_state_absent(self, checkpoint_manager):
        result = checkpoint_manager.load_trainer_state()
        assert result is None

    def test_invalid_run_name(self, test_system_cfg):
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=invalid_name, system_cfg=test_system_cfg)
