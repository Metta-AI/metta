"""Test the LRU cache functionality in CheckpointManager."""

import tempfile
from unittest.mock import patch

import pytest
import torch

from metta.agent.mocks import MockAgent
from metta.rl.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_run_dir):
    """Create a CheckpointManager with caching enabled."""
    return CheckpointManager(run_dir=temp_run_dir, run_name="test_run", cache_size=3)


@pytest.fixture
def checkpoint_manager_no_cache(temp_run_dir):
    """Create a CheckpointManager with caching disabled."""
    return CheckpointManager(run_dir=temp_run_dir, run_name="test_run", cache_size=0)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MockAgent()


class TestCheckpointCaching:
    """Test the LRU caching functionality."""

    def test_cache_hit_on_repeated_load(self, checkpoint_manager, mock_agent):
        """Test that loading the same checkpoint twice uses cache."""
        # Save a checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # First load - should load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent1 = checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1

        # Second load - should use cache
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent2 = checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 0  # Should not call torch.load
            assert agent1 is agent2  # Should be the same object from cache

    def test_cache_eviction_lru(self, checkpoint_manager, mock_agent):
        """Test that cache evicts least recently used items."""
        # Save 4 checkpoints (cache size is 3)
        for epoch in range(1, 5):
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata={"agent_step": epoch * 100})

        # Load all 4 checkpoints
        agents = []
        for epoch in range(1, 5):
            agents.append(checkpoint_manager.load_agent(epoch=epoch))

        # Cache should contain epochs 2, 3, 4 (epoch 1 was evicted)
        assert len(checkpoint_manager._cache) == 3

        # Load epoch 1 again - should load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1

        # Now cache should contain epochs 3, 4, 1 (epoch 2 was evicted)
        # Load epoch 3 - should still be in cache
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            checkpoint_manager.load_agent(epoch=3)
            assert mock_load.call_count == 0

    def test_cache_disabled(self, checkpoint_manager_no_cache, mock_agent):
        """Test that caching can be disabled by setting cache_size=0."""
        # Save a checkpoint
        checkpoint_manager_no_cache.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # Load twice - should always load from disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            checkpoint_manager_no_cache.load_agent(epoch=1)
            checkpoint_manager_no_cache.load_agent(epoch=1)
            assert mock_load.call_count == 2  # Both loads should hit disk

        # Cache should be empty
        assert len(checkpoint_manager_no_cache._cache) == 0

    def test_cache_invalidation_on_save(self, checkpoint_manager, mock_agent):
        """Test that cache is invalidated when a checkpoint is overwritten."""
        # Save and load a checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})
        checkpoint_manager.load_agent(epoch=1)  # Load to populate cache

        # Modify the agent
        mock_agent.dummy_value = 42

        # Save the same epoch again (overwrite)
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 200})

        # Load should get the new version from disk, not the cached old version
        agent2 = checkpoint_manager.load_agent(epoch=1)
        assert hasattr(agent2, "dummy_value") and agent2.dummy_value == 42

    def test_clear_cache(self, checkpoint_manager, mock_agent):
        """Test that clear_cache removes all cached items."""
        # Save and load multiple checkpoints
        for epoch in range(1, 4):
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata={"agent_step": epoch * 100})
            checkpoint_manager.load_agent(epoch=epoch)

        # Cache should have 3 items
        assert len(checkpoint_manager._cache) == 3

        # Clear cache
        checkpoint_manager.clear_cache()
        assert len(checkpoint_manager._cache) == 0

        # Next load should hit disk
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            checkpoint_manager.load_agent(epoch=1)
            assert mock_load.call_count == 1

    def test_cache_with_latest_epoch_load(self, checkpoint_manager, mock_agent):
        """Test caching works when loading latest epoch (epoch=None)."""
        # Save multiple checkpoints
        for epoch in range(1, 4):
            checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata={"agent_step": epoch * 100})

        # Load latest (should be epoch 3)
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent1 = checkpoint_manager.load_agent(epoch=None)
            assert mock_load.call_count == 1

        # Load latest again - should use cache
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent2 = checkpoint_manager.load_agent(epoch=None)
            assert mock_load.call_count == 0
            assert agent1 is agent2

        # Explicitly load epoch 3 - should also use cache
        with patch.object(torch, "load", wraps=torch.load) as mock_load:
            agent3 = checkpoint_manager.load_agent(epoch=3)
            assert mock_load.call_count == 0
            assert agent1 is agent3

    def test_cache_memory_efficiency(self, checkpoint_manager, mock_agent):
        """Test that cache maintains reference to same object."""
        # Save checkpoint
        checkpoint_manager.save_agent(mock_agent, epoch=1, metadata={"agent_step": 100})

        # Load multiple times
        agents = []
        for _ in range(5):
            agents.append(checkpoint_manager.load_agent(epoch=1))

        # All should be the same object reference
        assert all(a is agents[0] for a in agents)
