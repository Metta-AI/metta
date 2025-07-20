"""
Tests for PolicyCache using pytest.
"""

import threading
import time

import pytest

from metta.agent.mocks import MockPolicy
from metta.agent.policy_cache import PolicyCache
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord

# Create a minimal policy class that mimics MettaAgent structure


@pytest.fixture
def cache():
    """Create a cache instance for testing."""
    return PolicyCache(max_size=3)


@pytest.fixture
def policy_record_with_model():
    """Create a PolicyRecord with a loaded model."""
    metadata = PolicyMetadata(epoch=1, agent_step=100, generation=0, train_time=0.5)
    pr = PolicyRecord(None, "test_run", "file://test.pt", metadata)
    pr._cached_policy = MockPolicy()
    return pr


@pytest.fixture
def policy_record_metadata_only():
    """Create a PolicyRecord without a loaded model (metadata only)."""
    metadata = PolicyMetadata(epoch=2, agent_step=200, generation=1, train_time=1.0)
    pr = PolicyRecord(None, "metadata_only", "file://metadata.pt", metadata)
    # No _cached_policy set
    return pr


def create_policy_record(name: str, uri: str, with_model: bool = True) -> PolicyRecord:
    """Helper to create policy records for tests."""
    metadata = PolicyMetadata(
        epoch=int(name[-1]) if name[-1].isdigit() else 0, agent_step=100, generation=0, train_time=0.5
    )
    pr = PolicyRecord(None, name, uri, metadata)
    if with_model:
        pr._cached_policy = MockPolicy()
    return pr


class TestPolicyCacheBasics:
    """Test basic cache operations."""

    def test_init_with_invalid_size(self):
        """Test that cache initialization fails with invalid size."""
        with pytest.raises(ValueError, match="Cache size must be positive"):
            PolicyCache(max_size=0)

        with pytest.raises(ValueError, match="Cache size must be positive"):
            PolicyCache(max_size=-1)

    def test_put_and_get(self, cache, policy_record_with_model):
        """Test basic put and get operations."""
        # Put a record
        cache.put("key1", policy_record_with_model)

        # Get should return the same record
        retrieved = cache.get("key1")
        assert retrieved is policy_record_with_model

    def test_get_nonexistent(self, cache):
        """Test getting a non-existent key returns None."""
        assert cache.get("nonexistent") is None

    def test_update_existing(self, cache):
        """Test updating an existing key."""
        pr1 = create_policy_record("model1", "file://model1.pt")
        pr2 = create_policy_record("model1_updated", "file://model1.pt")

        cache.put("key1", pr1)
        cache.put("key1", pr2)  # Same key, different record

        assert cache.get("key1") is pr2


class TestPolicyCacheLRU:
    """Test LRU eviction behavior."""

    def test_lru_eviction(self, cache):
        """Test that least recently used items are evicted."""
        # Fill cache to capacity
        pr1 = create_policy_record("model1", "file://model1.pt")
        pr2 = create_policy_record("model2", "file://model2.pt")
        pr3 = create_policy_record("model3", "file://model3.pt")

        cache.put("model1", pr1)
        cache.put("model2", pr2)
        cache.put("model3", pr3)

        # Add one more, should evict model1 (least recently used)
        pr4 = create_policy_record("model4", "file://model4.pt")
        cache.put("model4", pr4)

        assert cache.get("model1") is None  # Evicted
        assert cache.get("model2") is not None
        assert cache.get("model3") is not None
        assert cache.get("model4") is not None

    def test_lru_with_access_pattern(self, cache):
        """Test LRU with specific access pattern."""
        # Fill cache
        pr1 = create_policy_record("model1", "file://model1.pt")
        pr2 = create_policy_record("model2", "file://model2.pt")
        pr3 = create_policy_record("model3", "file://model3.pt")

        cache.put("model1", pr1)
        cache.put("model2", pr2)
        cache.put("model3", pr3)

        # Access model1 and model3 to make them recently used
        cache.get("model1")
        cache.get("model3")

        # Add model4, should evict model2 (least recently used)
        pr4 = create_policy_record("model4", "file://model4.pt")
        cache.put("model4", pr4)

        assert cache.get("model1") is not None
        assert cache.get("model2") is None  # Evicted
        assert cache.get("model3") is not None
        assert cache.get("model4") is not None


class TestPolicyCacheConcurrency:
    """Test thread safety."""

    def test_concurrent_puts(self):
        """Test concurrent put operations."""
        cache = PolicyCache(max_size=50)
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    pr = create_policy_record(f"w{worker_id}_m{i}", f"file://w{worker_id}_m{i}.pt")
                    cache.put(f"w{worker_id}_m{i}", pr)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed read/write operations."""
        cache = PolicyCache(max_size=20)
        errors = []

        # Pre-populate some entries
        for i in range(10):
            pr = create_policy_record(f"initial{i}", f"file://initial{i}.pt")
            cache.put(f"initial{i}", pr)

        def worker(worker_id):
            try:
                for i in range(20):
                    # Mix of operations
                    if i % 2 == 0:
                        # Put
                        pr = create_policy_record(f"w{worker_id}_m{i}", f"file://w{worker_id}_m{i}.pt")
                        cache.put(f"w{worker_id}_m{i}", pr)
                    else:
                        # Get
                        cache.get(f"initial{i % 10}")

                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
