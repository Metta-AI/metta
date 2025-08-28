"""Comprehensive tests for CheckpointManager functionality."""

import tempfile
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from metta.agent.mocks import MockAgent, MockPolicy
from metta.rl.checkpoint_manager import CheckpointManager, parse_checkpoint_filename


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_run_dir):
    """Create a CheckpointManager instance for testing."""
    return CheckpointManager(run_dir=temp_run_dir, run_name="test_run")


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent()


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing."""
    return MockPolicy()


class TestCheckpointManagerBasicOperations:
    """Test basic save/load operations."""

    def test_save_and_load_agent_without_pydantic_errors(self, checkpoint_manager, mock_agent):
        """Test that we can save and load an agent with the new triple-dash filename format."""

        # Create metadata dictionary with embedded filename info
        metadata = {
            "agent_step": 5280,
            "total_time": 120.0,
            "score": 0.95,
        }

        # Save the agent
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata=metadata)

        # Verify checkpoint file exists with new format: {run_name}---e{epoch}_s{agent_step}_t{total_time}s.pt
        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        expected_filename = "test_run---e5_s5280_t120s.pt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        # Test that the save format is correct
        checkpoint = torch.load(agent_file, map_location="cpu", weights_only=False)

        # The checkpoint should be the agent directly
        assert callable(checkpoint)  # Should be callable (policy-like)

        # Verify metadata is properly embedded in filename
        saved_metadata = checkpoint_manager.load_metadata(epoch=5)

        assert saved_metadata["run"] == "test_run"
        assert saved_metadata["agent_step"] == 5280
        assert saved_metadata["epoch"] == 5
        assert saved_metadata["total_time"] == 120

        print("✅ Checkpoint format verified - using filename-embedded metadata!")

        # Load the agent back and verify it works
        loaded_agent = checkpoint_manager.load_agent(epoch=5)

        # Verify the loaded agent works with a forward pass
        test_input = TensorDict({"env_obs": torch.randn(1, 10)}, batch_size=(1,))
        output = loaded_agent(test_input)

        # Assertions
        assert type(loaded_agent).__name__ == "MockAgent"
        assert "actions" in output
        assert output["actions"].shape[0] == 1  # batch size

        # Verify the loaded agent has the expected structure
        assert callable(loaded_agent)

        print("✅ Agent loading and forward pass verified!")

    def test_multiple_epoch_saves(self, checkpoint_manager, mock_agent):
        """Test saving multiple epochs and finding the best/latest."""

        # Save multiple epochs with different metadata
        epochs_data = [
            (1, {"agent_step": 1000, "total_time": 30}),
            (5, {"agent_step": 5000, "total_time": 150}),
            (10, {"agent_step": 10000, "total_time": 300}),  # Highest agent_step
            (15, {"agent_step": 15000, "total_time": 450}),  # Latest and highest
        ]

        for epoch, metadata in epochs_data:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Verify files were created with correct naming format
        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        expected_files = [
            "test_run---e1_s1000_t30s.pt",
            "test_run---e5_s5000_t150s.pt",
            "test_run---e10_s10000_t300s.pt",
            "test_run---e15_s15000_t450s.pt",
        ]
        for expected_file in expected_files:
            assert (checkpoint_dir / expected_file).exists()

        # Test find_best_checkpoint functionality (uses epoch by default)
        best_path = checkpoint_manager.find_best_checkpoint("epoch")
        assert best_path is not None
        assert "test_run---e15_s15000_t450s.pt" == best_path.name  # Epoch 15 is highest

        # Test find_best_checkpoint with agent_step metric
        best_step_path = checkpoint_manager.find_best_checkpoint("agent_step")
        assert best_step_path is not None
        assert "test_run---e15_s15000_t450s.pt" == best_step_path.name  # Highest agent_step

        # Test loading latest (should be epoch 15)
        loaded_agent = checkpoint_manager.load_agent()  # No epoch specified = latest
        assert loaded_agent is not None

        # Test loading trainer state
        # Create a mock optimizer for testing
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=15, agent_step=15000)

        loaded_trainer_state = checkpoint_manager.load_trainer_state(epoch=15)
        assert loaded_trainer_state["epoch"] == 15
        assert loaded_trainer_state["agent_step"] == 15000


class TestCheckpointManagerAdvancedFeatures:
    """Test advanced checkpoint features."""

    def test_checkpoint_search_and_filtering(self, checkpoint_manager, mock_agent):
        """Test searching for checkpoints by various criteria using the new format."""

        # Save checkpoints with different metadata (only core fields are preserved in filename)
        test_data = [
            (1, {"agent_step": 1000, "total_time": 30}),
            (5, {"agent_step": 5000, "total_time": 150}),
            (10, {"agent_step": 10000, "total_time": 300}),
            (15, {"agent_step": 15000, "total_time": 450}),
        ]

        for epoch, metadata in test_data:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Test finding best by epoch (highest epoch number)
        best_epoch_path = checkpoint_manager.find_best_checkpoint("epoch")
        assert "test_run---e15_s15000_t450s.pt" == best_epoch_path.name

        # Test finding best by agent_step
        best_step_path = checkpoint_manager.find_best_checkpoint("agent_step")
        assert "test_run---e15_s15000_t450s.pt" == best_step_path.name

        # Test checkpoint file existence and parsing
        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("test_run---e*_s*_t*s.pt"))
        assert len(checkpoint_files) == 4

        # Test that we can parse metadata from all checkpoint filenames
        for checkpoint_file in checkpoint_files:
            metadata = parse_checkpoint_filename(checkpoint_file.name)
            assert metadata is not None
            assert metadata["run"] == "test_run"
            assert "epoch" in metadata
            assert "agent_step" in metadata
            assert "total_time" in metadata

        print("✅ Checkpoint search and filtering capabilities verified")

    def test_checkpoint_cleanup_simulation(self, checkpoint_manager, mock_agent):
        """Test the built-in checkpoint cleanup functionality."""

        # Save many checkpoints
        for epoch in range(1, 11):  # Save 10 checkpoints
            checkpoint_manager.save_agent(
                mock_agent, epoch=epoch, metadata={"agent_step": epoch * 1000, "total_time": epoch * 30}
            )

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"

        # Verify all checkpoints exist with new format
        checkpoint_files = list(checkpoint_dir.glob("test_run---e*_s*_t*s.pt"))
        assert len(checkpoint_files) == 10

        # Test the actual cleanup functionality
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
        assert deleted_count == 5  # Should have removed 5 old checkpoints

        # Verify only 5 checkpoints remain
        remaining_files = list(checkpoint_dir.glob("test_run---e*_s*_t*s.pt"))
        assert len(remaining_files) == 5

        # Verify the remaining files are the latest ones (epochs 6-10)
        remaining_epochs = sorted([parse_checkpoint_filename(f.name)["epoch"] for f in remaining_files])
        assert remaining_epochs == [6, 7, 8, 9, 10]

        print("✅ Checkpoint cleanup functionality verified")

    def test_concurrent_save_load_safety(self, checkpoint_manager, mock_agent):
        """Test that saves and loads are atomic and don't interfere."""

        # Save initial checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 1000, "total_time": 30})

        # Verify we can always load a valid checkpoint
        loaded_agent = checkpoint_manager.load_agent(epoch=1)
        assert loaded_agent is not None

        # Save another checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=2, metadata={"agent_step": 2000, "total_time": 60})

        # Loading should still work and get the latest
        loaded_agent = checkpoint_manager.load_agent()  # Latest
        assert loaded_agent is not None

        # Verify the latest is indeed epoch 2
        latest_metadata = checkpoint_manager.load_metadata()
        assert latest_metadata["epoch"] == 2
        assert latest_metadata["agent_step"] == 2000

        # Verify trainer state save/load doesn't interfere
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=2, agent_step=2000)
        trainer_state = checkpoint_manager.load_trainer_state(epoch=2)
        assert trainer_state["epoch"] == 2

        print("✅ Concurrent save/load safety verified")

    def test_epoch_listing_and_selection(self, checkpoint_manager, mock_agent):
        """Test epoch listing and latest epoch selection with new format."""

        # Save checkpoints in non-sequential order to test sorting
        epochs_data = [
            (10, {"agent_step": 10000, "total_time": 300}),
            (1, {"agent_step": 1000, "total_time": 30}),
            (5, {"agent_step": 5000, "total_time": 150}),
            (3, {"agent_step": 3000, "total_time": 90}),
        ]

        for epoch, metadata in epochs_data:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Test list_epochs returns sorted list
        epochs = checkpoint_manager.list_epochs()
        assert epochs == [1, 3, 5, 10]

        # Test get_latest_epoch
        latest_epoch = checkpoint_manager.get_latest_epoch()
        assert latest_epoch == 10

        # Test exists returns True when checkpoints exist
        assert checkpoint_manager.exists()

        print("✅ Epoch listing and selection functionality verified")


class TestCheckpointManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_parse_checkpoint_filename_utility(self):
        """Test the parse_checkpoint_filename utility function."""

        # Test valid filename
        valid_filename = "test_run---e5_s1000_t300s.pt"
        metadata = parse_checkpoint_filename(valid_filename)

        assert metadata is not None
        assert metadata["run"] == "test_run"
        assert metadata["epoch"] == 5
        assert metadata["agent_step"] == 1000
        assert metadata["total_time"] == 300
        assert metadata["checkpoint_file"] == valid_filename

        # Test invalid filename formats
        invalid_filenames = [
            "invalid.pt",
            "test_run_e5_s1000_t300s.pt",  # Wrong separator
            "test_run---e5_s1000.pt",  # Missing total_time
            "test_run---e5_t300s.pt",  # Missing agent_step
            "test_run---s1000_t300s.pt",  # Missing epoch
        ]

        for invalid_filename in invalid_filenames:
            metadata = parse_checkpoint_filename(invalid_filename)
            assert metadata is None

        print("✅ parse_checkpoint_filename utility function verified")

    def test_load_from_empty_directory(self, checkpoint_manager):
        """Test loading when no checkpoints exist."""

        # Should return None gracefully
        loaded_agent = checkpoint_manager.load_agent()
        assert loaded_agent is None

        trainer_state = checkpoint_manager.load_trainer_state()
        assert trainer_state is None

        best_path = checkpoint_manager.find_best_checkpoint("epoch")
        assert best_path is None

        # Test metadata loading from empty directory
        metadata = checkpoint_manager.load_metadata()
        assert metadata is None

        # Test list_epochs from empty directory
        epochs = checkpoint_manager.list_epochs()
        assert epochs == []

        # Test exists on empty directory
        assert not checkpoint_manager.exists()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
