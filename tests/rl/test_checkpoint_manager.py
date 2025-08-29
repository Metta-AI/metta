"""Consolidated tests for CheckpointManager core functionality.

Tests basic save/load operations, caching, trainer state management,
cleanup operations, filename parsing utilities, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from metta.agent.mocks import MockAgent
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
def cached_checkpoint_manager(temp_run_dir):
    """Create a CheckpointManager with caching enabled."""
    return CheckpointManager(run_dir=temp_run_dir, run_name="test_run", cache_size=3)


@pytest.fixture
def no_cache_checkpoint_manager(temp_run_dir):
    """Create a CheckpointManager with caching disabled."""
    return CheckpointManager(run_dir=temp_run_dir, run_name="test_run", cache_size=0)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent()


class TestBasicSaveLoad:
    """Test basic save/load operations."""

    def test_save_and_load_agent(self, checkpoint_manager, mock_agent):
        """Test basic save and load functionality with filename metadata."""
        # Create metadata dictionary
        metadata = {"agent_step": 5280, "total_time": 120.0, "score": 0.75}

        # Save the agent
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata=metadata)

        # Verify checkpoint file exists with correct format
        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        expected_filename = "test_run.e5.s5280.t120.sc7500.pt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        # Verify metadata is embedded in filename
        parsed = parse_checkpoint_filename(expected_filename)
        assert parsed[0] == "test_run"  # run name
        assert parsed[1] == 5  # epoch
        assert parsed[2] == 5280  # agent_step
        assert parsed[3] == 120  # total_time
        assert abs(parsed[4] - 0.75) < 0.0001  # score

        # Load the agent back
        loaded_agent = checkpoint_manager.load_agent(epoch=5)
        assert loaded_agent is not None

        # Verify the loaded agent works
        test_input = TensorDict({"env_obs": torch.randn(1, 10)}, batch_size=(1,))
        output = loaded_agent(test_input)
        assert "actions" in output
        assert output["actions"].shape[0] == 1

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent):
        """Test saving multiple epochs and selecting latest/best."""
        epochs_data = [
            (1, {"agent_step": 1000, "total_time": 30, "score": 0.5}),
            (5, {"agent_step": 5000, "total_time": 150, "score": 0.8}),
            (10, {"agent_step": 10000, "total_time": 300, "score": 0.9}),
        ]

        for epoch, metadata in epochs_data:
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Test loading latest (should be epoch 10)
        loaded_agent = checkpoint_manager.load_agent()
        assert loaded_agent is not None

        # Test checkpoint selection
        latest_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1, metric="epoch")
        assert len(latest_checkpoints) == 1
        assert "test_run.e10.s10000.t300.sc9000.pt" == latest_checkpoints[0].name

        # Test selection by score
        best_score_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1, metric="score")
        assert len(best_score_checkpoints) == 1
        assert "test_run.e10.s10000.t300.sc9000.pt" == best_score_checkpoints[0].name

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent):
        """Test trainer state persistence."""
        # Save agent checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata={"agent_step": 1000, "total_time": 60})

        # Create and save trainer state
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        stopwatch_state = {"elapsed_time": 123.45}
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=5, agent_step=1000, stopwatch_state=stopwatch_state)

        # Load trainer state
        loaded_trainer_state = checkpoint_manager.load_trainer_state(epoch=5)
        assert loaded_trainer_state is not None
        assert loaded_trainer_state["epoch"] == 5
        assert loaded_trainer_state["agent_step"] == 1000
        assert loaded_trainer_state["stopwatch_state"]["elapsed_time"] == 123.45
        assert "optimizer_state" in loaded_trainer_state


class TestCaching:
    """Test LRU caching functionality."""

    def test_cache_hit_on_repeated_load(self, cached_checkpoint_manager, mock_agent):
        """Test that loading the same checkpoint twice uses cache."""
        # Save a checkpoint
        cached_checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # First load - should load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent1 = cached_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1

            # Second load - should use cache
            agent2 = cached_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1  # Still 1, used cache
            assert agent1 is agent2  # Same object reference

    def test_cache_eviction_lru(self, cached_checkpoint_manager, mock_agent):
        """Test LRU cache eviction when cache limit exceeded."""
        # Save 4 checkpoints (cache size is 3)
        for epoch in range(1, 5):
            cached_checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata={"agent_step": epoch * 100})

        # Load all 4 - should evict oldest when loading 4th
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            cached_checkpoint_manager.load_agent(epoch=1)  # Cache: [1]
            cached_checkpoint_manager.load_agent(epoch=2)  # Cache: [1, 2]
            cached_checkpoint_manager.load_agent(epoch=3)  # Cache: [1, 2, 3]
            cached_checkpoint_manager.load_agent(epoch=4)  # Cache: [2, 3, 4] (evicted 1)

            assert mock_load.call_count == 4

            # Load epoch 1 again - should reload from disk (was evicted)
            cached_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 5

            # Load epoch 2 again - should use cache
            cached_checkpoint_manager.load_agent(epoch=2)
            assert mock_load.call_count == 5  # Still 5, used cache

    def test_cache_disabled(self, no_cache_checkpoint_manager, mock_agent):
        """Test that caching can be disabled."""
        # Save checkpoint
        no_cache_checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # Load multiple times - should always load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            no_cache_checkpoint_manager.load_agent(epoch=1)
            no_cache_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 2  # Loaded twice

    def test_cache_invalidation_on_save(self, cached_checkpoint_manager, mock_agent):
        """Test that cache is invalidated when checkpoint is overwritten."""
        # Save and load to populate cache
        cached_checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})
        cached_checkpoint_manager.load_agent(epoch=1)

        # Save over same epoch (should invalidate cache)
        cached_checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 200})

        # Load again - should reload from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            cached_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1


class TestCleanup:
    """Test checkpoint cleanup functionality."""

    def test_cleanup_old_checkpoints(self, checkpoint_manager, mock_agent):
        """Test cleanup functionality."""
        # Save 10 checkpoints
        for epoch in range(1, 11):
            checkpoint_manager.save_agent(
                mock_agent, epoch=epoch, metadata={"agent_step": epoch * 1000, "total_time": epoch * 30}
            )

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("test_run.e*.s*.t*.sc*.pt"))
        assert len(checkpoint_files) == 10

        # Clean up, keeping only 5
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
        assert deleted_count == 5

        # Verify only 5 remain (latest ones: epochs 6-10)
        remaining_files = list(checkpoint_dir.glob("test_run.e*.s*.t*.sc*.pt"))
        assert len(remaining_files) == 5

        remaining_epochs = sorted([parse_checkpoint_filename(f.name)[1] for f in remaining_files])
        assert remaining_epochs == [6, 7, 8, 9, 10]

    def test_cleanup_with_trainer_state(self, checkpoint_manager, mock_agent):
        """Test that cleanup removes both agent and trainer state files."""
        # Save checkpoint and trainer state
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 1000, "total_time": 60})
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=1, agent_step=1000)

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        assert (checkpoint_dir / "test_run.e1.s1000.t60.sc0.pt").exists()
        assert (checkpoint_dir / "test_run.e1.trainer.pt").exists()

        # Cleanup should remove both
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=0)
        assert deleted_count == 1
        assert not (checkpoint_dir / "test_run.e1.s1000.t60.sc0.pt").exists()
        assert not (checkpoint_dir / "test_run.e1.trainer.pt").exists()


class TestUtilities:
    """Test utility functions."""

    def test_parse_checkpoint_filename_valid(self):
        """Test parsing valid checkpoint filenames."""
        filename = "my_run.e42.s12500.t1800.sc8750.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("my_run", 42, 12500, 1800, 0.8750)

        # Test edge cases
        filename = "run.e0.s0.t0.sc0.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 0, 0, 0, 0.0)

        filename = "run.e999.s999999.t86400.sc9999.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 999, 999999, 86400, 0.9999)

    def test_parse_checkpoint_filename_invalid(self):
        """Test parsing invalid checkpoint filenames."""
        invalid_filenames = [
            "invalid.pt",
            "run_e5_s1000_t300.pt",  # Wrong separators
            "run.e5.s1000.pt",  # Missing fields
            "run.epoch5.s1000.t300.sc0.pt",  # Wrong prefixes
            "run.e5.s1000.t300.sc0.txt",  # Wrong extension
        ]

        for invalid_filename in invalid_filenames:
            with pytest.raises(ValueError):
                parse_checkpoint_filename(invalid_filename)

    def test_exists_functionality(self, checkpoint_manager, mock_agent):
        """Test the exists() method."""
        # Should not exist initially
        assert not checkpoint_manager.exists()

        # Save a checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100, "total_time": 30})

        # Should exist now
        assert checkpoint_manager.exists()

    def test_get_checkpoint_uri(self, checkpoint_manager, mock_agent):
        """Test getting checkpoint URIs."""
        # Save checkpoints
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 1000, "total_time": 60})
        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata={"agent_step": 5000, "total_time": 300})

        # Get specific epoch URI
        uri = checkpoint_manager.get_checkpoint_uri(epoch=1)
        assert uri.startswith("file://")
        assert uri.endswith("test_run.e1.s1000.t60.sc0.pt")

        # Get latest epoch URI
        latest_uri = checkpoint_manager.get_checkpoint_uri()
        assert latest_uri.startswith("file://")
        assert latest_uri.endswith("test_run.e5.s5000.t300.sc0.pt")


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_load_from_empty_directory(self, checkpoint_manager):
        """Test loading when no checkpoints exist."""
        # Should return None when no checkpoints exist
        result = checkpoint_manager.load_agent()
        assert result is None

        result = checkpoint_manager.load_trainer_state()
        assert result is None

        # Should return empty list for selection
        checkpoints = checkpoint_manager.select_checkpoints()
        assert checkpoints == []

    def test_invalid_run_name(self, temp_run_dir):
        """Test that invalid run names are rejected."""
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run_dir=temp_run_dir, run_name=invalid_name)

    def test_missing_checkpoint_error(self, checkpoint_manager):
        """Test error when trying to get URI for non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.get_checkpoint_uri(epoch=999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
