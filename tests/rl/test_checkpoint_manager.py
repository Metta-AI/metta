import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from metta.agent.mocks import MockAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig


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


class TestBasicSaveLoad:
    def test_latest_selector_file_uri(self, checkpoint_manager, mock_agent):
        """Test :latest selector for file:// URIs."""
        # Save multiple checkpoints
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={})
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata={})
        checkpoint_manager.save_agent(mock_agent, epoch=3, metadata={})

        # Test :latest resolution
        from metta.rl.checkpoint_manager import key_and_version

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.pt"
        run_name, epoch = key_and_version(latest_uri)

        assert run_name == "test_run"
        assert epoch == 5  # Should resolve to highest epoch

    def test_load_from_uri_with_latest(self, checkpoint_manager, mock_agent):
        """Test loading policy with :latest selector."""
        # Save multiple checkpoints
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={})
        checkpoint_manager.save_agent(mock_agent, epoch=7, metadata={})
        checkpoint_manager.save_agent(mock_agent, epoch=3, metadata={})

        # Load using :latest selector
        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.pt"
        loaded_agent = CheckpointManager.load_from_uri(latest_uri)

        assert loaded_agent is not None
        # Verify it loaded the correct checkpoint by checking metadata
        metadata = CheckpointManager.get_policy_metadata(latest_uri)
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 7  # Should be the highest epoch

    def test_save_and_load_agent(self, checkpoint_manager, mock_agent):
        metadata = {"agent_step": 5280, "total_time": 120.0, "score": 0.75}

        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata=metadata)

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        expected_filename = "test_run:v5.pt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        metadata = CheckpointManager.get_policy_metadata(agent_file.as_uri())
        assert "run_name" in metadata and metadata["run_name"] == "test_run"
        assert "epoch" in metadata and metadata["epoch"] == 5

        loaded_agent = CheckpointManager.load_from_uri(agent_file.as_uri())
        assert loaded_agent is not None

        test_input = TensorDict({"env_obs": torch.randn(1, 10)}, batch_size=(1,))
        output = loaded_agent(test_input)
        assert "actions" in output
        assert output["actions"].shape[0] == 1

    def test_remote_prefix_upload(self, test_system_cfg, mock_agent):
        metadata = {"agent_step": 123, "total_time": 10, "score": 0.5}

        test_system_cfg.local_only = False
        test_system_cfg.remote_prefix = "s3://bucket/checkpoints"
        manager = CheckpointManager(run="test_run", system_cfg=test_system_cfg)

        expected_filename = "test_run:v3.pt"
        expected_remote = f"s3://bucket/checkpoints/{expected_filename}"

        with patch("metta.rl.checkpoint_manager.write_file") as mock_write:
            remote_uri = manager.save_agent(mock_agent, epoch=3, metadata=metadata)

        assert remote_uri == expected_remote
        mock_write.assert_called_once()
        remote_arg, local_arg = mock_write.call_args[0]
        assert remote_arg == expected_remote
        assert Path(local_arg).name == expected_filename

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent):
        epochs_data = [
            (1, {"agent_step": 1000, "total_time": 30, "score": 0.5}),
            (5, {"agent_step": 5000, "total_time": 150, "score": 0.8}),
            (10, {"agent_step": 10000, "total_time": 300, "score": 0.9}),
        ]

        for epoch, metadata in epochs_data:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Test checkpoint selection and loading of latest (should be epoch 10)
        latest_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1)
        assert len(latest_checkpoints) == 1
        assert latest_checkpoints[0].endswith("test_run:v10.pt")
        loaded_agent = CheckpointManager.load_from_uri(latest_checkpoints[0])
        assert loaded_agent is not None

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent):
        # Save agent checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata={"agent_step": 1000, "total_time": 60})

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

    def test_checkpoint_existence(self, checkpoint_manager, mock_agent):
        # Should raise FileNotFoundError when no checkpoints exist
        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.pt"
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri(latest_uri)

        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100, "total_time": 30})
        saved_uri = checkpoint_manager.select_checkpoints()[0]
        loaded = CheckpointManager.load_from_uri(saved_uri)
        assert loaded is not None


class TestCleanup:
    def test_cleanup_old_checkpoints(self, checkpoint_manager, mock_agent):
        # Save 10 checkpoints
        for epoch in range(1, 11):
            checkpoint_manager.save_agent(
                mock_agent, epoch=epoch, metadata={"agent_step": epoch * 1000, "total_time": epoch * 30}
            )

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        checkpoint_files = [p for p in checkpoint_dir.glob("*.pt") if ":v" in p.stem]
        assert len(checkpoint_files) == 10

        # Clean up, keeping only 5
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
        assert deleted_count == 5

        # Verify only 5 remain (latest ones: epochs 6-10)
        remaining_files = [p for p in checkpoint_dir.glob("*.pt") if ":v" in p.stem]
        assert len(remaining_files) == 5

        remaining_epochs = sorted(int(f.stem.split(":v")[1]) for f in remaining_files)
        assert remaining_epochs == [6, 7, 8, 9, 10]

    def test_cleanup_with_trainer_state(self, checkpoint_manager, mock_agent):
        # Save checkpoint and trainer state
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 1000, "total_time": 60})
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=1, agent_step=1000)

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        assert (checkpoint_dir / "test_run:v1.pt").exists()
        assert (checkpoint_dir / "trainer_state.pt").exists()

        # Cleanup should remove both
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=0)
        assert deleted_count == 1
        assert not (checkpoint_dir / "test_run:v1.pt").exists()
        assert not (checkpoint_dir / "trainer_state.pt").exists()


class TestErrorHandling:
    def test_load_from_empty_directory(self, checkpoint_manager):
        # Should raise FileNotFoundError when no checkpoints exist
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri(f"file://{checkpoint_manager.checkpoint_dir}/test_run:latest.pt")

        # Trainer state should return None when not found
        result = checkpoint_manager.load_trainer_state()
        assert result is None

        # Should return empty list for selection
        checkpoints = checkpoint_manager.select_checkpoints()
        assert checkpoints == []

    def test_invalid_run_name(self, test_system_cfg):
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=invalid_name, system_cfg=test_system_cfg)
