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
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_run_dir):
    return CheckpointManager(run="test_run", run_dir=temp_run_dir)


@pytest.fixture
def cached_checkpoint_manager(temp_run_dir):
    return CheckpointManager(run="test_run", run_dir=temp_run_dir, cache_size=3)


@pytest.fixture
def no_cache_checkpoint_manager(temp_run_dir):
    return CheckpointManager(run="test_run", run_dir=temp_run_dir, cache_size=0)


@pytest.fixture
def mock_agent():
    return MockAgent()


class TestBasicSaveLoad:
    def test_save_and_load_agent(self, checkpoint_manager, mock_agent):
        metadata = {"agent_step": 5280, "total_time": 120.0, "score": 0.75}

        checkpoint_manager.save_agent(mock_agent, epoch=5, metadata=metadata)

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        expected_filename = "test_run__e5__s5280__t120__sc7500.pt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        parsed = parse_checkpoint_filename(expected_filename)
        assert parsed[0] == "test_run"  # run name
        assert parsed[1] == 5  # epoch
        assert parsed[2] == 5280  # agent_step
        assert parsed[3] == 120  # total_time
        assert abs(parsed[4] - 0.75) < 0.0001  # score

        loaded_agent = checkpoint_manager.load_agent(epoch=5)
        assert loaded_agent is not None

        test_input = TensorDict({"env_obs": torch.randn(1, 10)}, batch_size=(1,))
        output = loaded_agent(test_input)
        assert "actions" in output
        assert output["actions"].shape[0] == 1

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent):
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
        assert latest_checkpoints[0].endswith("test_run__e10__s10000__t300__sc9000.pt")

        # Test selection by score
        best_score_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1, metric="score")
        assert len(best_score_checkpoints) == 1
        assert best_score_checkpoints[0].endswith("test_run__e10__s10000__t300__sc9000.pt")

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
        assert "optimizer_state" in loaded_trainer_state


class TestCaching:
    def test_cache_hit_on_repeated_load(self, cached_checkpoint_manager, mock_agent):
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

            # Load epoch 2 again - should reload from disk (was evicted when epoch 1 was reloaded)
            cached_checkpoint_manager.load_agent(epoch=2)
            assert mock_load.call_count == 6  # Reloaded epoch 2

    def test_cache_disabled(self, no_cache_checkpoint_manager, mock_agent):
        # Save checkpoint
        no_cache_checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # Load multiple times - should always load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            no_cache_checkpoint_manager.load_agent(epoch=1)
            no_cache_checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 2  # Loaded twice

    def test_cache_invalidation_on_save(self, cached_checkpoint_manager, mock_agent):
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
    def test_cleanup_old_checkpoints(self, checkpoint_manager, mock_agent):
        # Save 10 checkpoints
        for epoch in range(1, 11):
            checkpoint_manager.save_agent(
                mock_agent, epoch=epoch, metadata={"agent_step": epoch * 1000, "total_time": epoch * 30}
            )

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("test_run__e*__s*__t*__sc*.pt"))
        assert len(checkpoint_files) == 10

        # Clean up, keeping only 5
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
        assert deleted_count == 5

        # Verify only 5 remain (latest ones: epochs 6-10)
        remaining_files = list(checkpoint_dir.glob("test_run__e*__s*__t*__sc*.pt"))
        assert len(remaining_files) == 5

        remaining_epochs = sorted([parse_checkpoint_filename(f.name)[1] for f in remaining_files])
        assert remaining_epochs == [6, 7, 8, 9, 10]

    def test_cleanup_with_trainer_state(self, checkpoint_manager, mock_agent):
        # Save checkpoint and trainer state
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 1000, "total_time": 60})
        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=1, agent_step=1000)

        checkpoint_dir = Path(checkpoint_manager.run_dir) / "test_run" / "checkpoints"
        assert (checkpoint_dir / "test_run__e1__s1000__t60__sc0.pt").exists()
        assert (checkpoint_dir / "trainer_state.pt").exists()

        # Cleanup should remove both
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(keep_last_n=0)
        assert deleted_count == 1
        assert not (checkpoint_dir / "test_run__e1__s1000__t60__sc0.pt").exists()
        assert not (checkpoint_dir / "trainer_state.pt").exists()


class TestUtilities:
    def test_parse_checkpoint_filename_valid(self):
        filename = "my_run__e42__s12500__t1800__sc8750.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("my_run", 42, 12500, 1800, 0.8750)

        # Test edge cases
        filename = "run__e0__s0__t0__sc0.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 0, 0, 0, 0.0)

        filename = "run__e999__s999999__t86400__sc9999.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 999, 999999, 86400, 0.9999)

    def test_parse_checkpoint_filename_invalid(self):
        invalid_filenames = [
            "invalid.pt",
            "run_e5_s1000_t300.pt",  # Wrong separators
            "run__e5__s1000.pt",  # Missing fields
            "run__epoch5__s1000__t300__sc0.pt",  # Wrong prefixes
            "run__e5__s1000__t300__sc0.txt",  # Wrong extension
        ]

        for invalid_filename in invalid_filenames:
            with pytest.raises(ValueError):
                parse_checkpoint_filename(invalid_filename)

    def test_checkpoint_existence(self, checkpoint_manager, mock_agent):
        # Should raise FileNotFoundError initially
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_agent()

        # Save a checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100, "total_time": 30})

        # Should load successfully now
        loaded = checkpoint_manager.load_agent()
        assert loaded is not None


class TestErrorHandling:
    def test_load_from_empty_directory(self, checkpoint_manager):
        # Should raise FileNotFoundError when no checkpoints exist
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_agent()

        # Trainer state should return None when not found
        result = checkpoint_manager.load_trainer_state()
        assert result is None

        # Should return empty list for selection
        checkpoints = checkpoint_manager.select_checkpoints()
        assert checkpoints == []

    def test_invalid_run_name(self, temp_run_dir):
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=invalid_name, run_dir=temp_run_dir)
