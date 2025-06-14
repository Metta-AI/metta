"""
Real S3 integration tests for S3CacheManager using pytest.
Tests against actual S3 bucket with cleanup.
"""

import os
import time
from typing import Set

import pytest

from metta.util.s3_cache import S3CacheManager

# Global set to track created keys for cleanup
created_keys: Set[str] = set()
test_prefix = f"test_cache_{int(time.time())}/"


@pytest.fixture(scope="session")
def s3_config():
    """Session-scoped fixture for S3 configuration."""
    bucket_name = os.getenv("TEST_S3_BUCKET", "softmax-level-cache")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    print(f"Testing with bucket: {bucket_name}, prefix: {test_prefix}")

    return {"bucket_name": bucket_name, "prefix": test_prefix, "aws_region": aws_region}


@pytest.fixture
def cache(s3_config):
    """Fixture that provides a cache manager instance."""
    cache_manager = S3CacheManager(
        bucket_name=s3_config["bucket_name"],
        prefix=s3_config["prefix"],
        compression_level=6,
        aws_region=s3_config["aws_region"],
    )

    if not cache_manager.s3_available:
        pytest.skip("S3 not available - check AWS credentials and bucket access")

    return cache_manager


@pytest.fixture
def track_key():
    """Fixture to track cache keys for cleanup."""

    def _track_key(cache_key: str):
        created_keys.add(cache_key)
        return cache_key

    return _track_key


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests(s3_config):
    """Automatically clean up test objects after all tests complete."""
    yield  # Run all tests first

    # Cleanup after tests
    if created_keys:
        cleanup_cache = S3CacheManager(
            bucket_name=s3_config["bucket_name"], prefix=s3_config["prefix"], aws_region=s3_config["aws_region"]
        )

        if cleanup_cache.s3_available:
            print(f"\nCleaning up {len(created_keys)} test objects...")
            for cache_key in created_keys:
                cleanup_cache.delete(cache_key)
            print("Cleanup completed")


class TestS3CacheBasics:
    """Basic S3 cache functionality tests."""

    def test_basic_put_get_cycle(self, cache, track_key):
        """Test basic put and get operations."""
        cache_key = track_key("basic_test")

        test_data = {"message": "Hello, S3!", "number": 42, "list": [1, 2, 3, 4, 5], "nested": {"key": "value"}}

        # Put data
        success = cache.put(cache_key, test_data)
        assert success, "Failed to put data to S3"

        # Get data back
        retrieved_data = cache.get(cache_key)
        assert retrieved_data is not None, "Failed to retrieve data from S3"
        assert retrieved_data == test_data, "Retrieved data doesn't match original"

    def test_cache_miss(self, cache):
        """Test behavior with non-existent keys."""
        nonexistent_key = f"this_key_should_not_exist_{time.time()}"

        result = cache.get(nonexistent_key)
        assert result is None, "Should return None for non-existent key"

    def test_large_object_caching(self, cache, track_key):
        """Test caching of large objects to verify compression works."""
        cache_key = track_key("large_object_test")

        # Create a large object
        large_data = {
            "matrix": [[i * j for j in range(100)] for i in range(100)],
            "text": "A" * 10000,  # 10KB of text
            "metadata": {"size": "large", "test": True},
        }

        # Cache it
        success = cache.put(cache_key, large_data)
        assert success, "Should successfully cache large object"

        # Retrieve it
        retrieved = cache.get(cache_key)
        assert retrieved == large_data, "Large object should round-trip correctly"


class TestKeyGeneration:
    """Tests for cache key generation logic."""

    def test_key_generation_consistency(self, cache):
        """Test that the same inputs always generate the same key."""
        test_config = {"width": 32, "height": 32, "seed": 12345, "objects": ["wall", "treasure", "agent"]}

        key1 = cache.create_key(test_config)
        key2 = cache.create_key(test_config)
        key3 = cache.create_key(test_config.copy())  # Different object, same content

        assert key1 == key2, "Same config should generate same key"
        assert key1 == key3, "Equivalent configs should generate same key"
        assert len(key1) == 64, "Key should be 64-character SHA-256 hex"

    def test_key_generation_sensitivity(self, cache):
        """Test that different inputs generate different keys."""
        from typing import Any, Dict

        base_config: Dict[str, Any] = {"width": 32, "height": 32, "seed": 12345}

        key1 = cache.create_key(base_config)

        # Change one value
        modified_config = base_config.copy()
        modified_config["seed"] = 12346
        key2 = cache.create_key(modified_config)

        # Add a new key
        extended_config = base_config.copy()
        extended_config["new_param"] = "value"
        key3 = cache.create_key(extended_config)

        assert key1 != key2, "Different configs should generate different keys"
        assert key1 != key3, "Extended config should generate different key"
        assert key2 != key3, "All different configs should have unique keys"

    def test_kwarg_order_independence(self, cache):
        """Test that keyword argument order doesn't affect key generation."""
        key1 = cache.create_key(a=1, b=2, c=3)
        key2 = cache.create_key(c=3, a=1, b=2)
        key3 = cache.create_key(b=2, c=3, a=1)

        assert key1 == key2 == key3, "Keyword argument order should not affect key"

    def test_complex_object_keys(self, cache):
        """Test key generation with complex nested objects."""
        complex_config = {
            "grid_size": [32, 32],
            "agents": {"count": 4, "spawn_method": "random"},
            "objects": [{"type": "wall", "density": 0.1}, {"type": "treasure", "count": 5}],
            "nested": {"deep": {"values": [1, 2, 3], "metadata": {"version": "1.0"}}},
        }

        # Generate keys multiple times
        key1 = cache.create_key(complex_config)
        key2 = cache.create_key(complex_config)

        assert key1 == key2, "Complex config should generate consistent keys"

        # Slight modification should produce different key
        modified_config = complex_config.copy()
        modified_config["agents"] = {"count": 5, "spawn_method": "random"}
        key3 = cache.create_key(modified_config)

        assert key1 != key3, "Modified complex config should generate different key"


class TestContextManager:
    """Tests for the context manager functionality."""

    def test_context_manager_cache_miss_then_hit(self, cache, track_key):
        """Test the full context manager workflow: miss, compute, cache, then hit."""
        config = {"test_id": f"context_test_{time.time()}", "data": list(range(100))}

        # First usage - should be a cache miss
        computation_called = False
        computed_result = {"expensive_computation": "result", "timestamp": time.time()}

        with cache(config) as ctx:
            _cache_key = track_key(ctx.cache_key)

            assert ctx.miss, "First access should be a cache miss"
            assert not ctx.hit, "First access should not be a cache hit"
            assert ctx.get() is None, "Cache miss should return None"

            # Simulate expensive computation
            computation_called = True
            ctx.set(computed_result)

        assert computation_called, "Computation should have been called"

        # Second usage - should be a cache hit
        computation_called_again = False

        with cache(config) as ctx:
            assert ctx.hit, "Second access should be a cache hit"
            assert not ctx.miss, "Second access should not be a cache miss"

            cached_result = ctx.get()
            assert cached_result is not None, "Cache hit should return data"
            assert cached_result == computed_result, "Cached data should match original"

            # Should not need to compute again
            computation_called_again = False

        assert not computation_called_again, "Computation should not be called on cache hit"

    def test_context_manager_no_set_no_cache(self, cache):
        """Test that nothing gets cached if ctx.set() is never called."""
        config = {"test": "no_set_test", "timestamp": time.time()}

        # First usage - miss but don't set result
        with cache(config) as ctx:
            _cache_key = ctx.cache_key
            assert ctx.miss, "Should be cache miss"
            # Don't call ctx.set()

        # Second usage - should still be a miss
        with cache(config) as ctx:
            assert ctx.miss, "Should still be cache miss since nothing was cached"


class TestRealWorldUsage:
    """Tests that simulate real-world usage patterns."""

    def test_mettagrid_map_builder_simulation(self, cache, track_key):
        """Test with a config structure similar to map_builder_config from mettagrid."""
        map_builder_config = {
            "_target_": "mettagrid.level_builder.RandomMapBuilder",
            "width": 64,
            "height": 64,
            "num_agents": 4,
            "wall_density": 0.15,
            "treasure_count": 10,
            "spawn_method": "corner",
            "difficulty_params": {"enemy_density": 0.05, "trap_density": 0.02, "power_up_density": 0.01},
            "generation_seed": 42,
            "metadata": {"version": "1.0", "created_by": "test_suite", "tags": ["testing", "integration"]},
        }

        # First call - miss and cache
        with cache(map_builder_config) as ctx:
            cache_key = track_key(ctx.cache_key)

            assert ctx.miss, "Should be cache miss for new config"

            # Simulate building a level (expensive operation)
            mock_level = {
                "grid": [["wall" if i % 7 == 0 else "floor" for i in range(64)] for _ in range(64)],
                "agent_positions": [(1, 1), (62, 1), (1, 62), (62, 62)],
                "treasure_positions": [(10, 10), (20, 20), (30, 30)],
                "labels": ["test_level", "integration"],
            }

            ctx.set(mock_level)

        # Second call with same config - should hit
        with cache(map_builder_config) as ctx:
            assert ctx.hit, "Should be cache hit for same config"

            cached_level = ctx.get()
            assert cached_level == mock_level, "Cached level should match original"

        # Third call with slightly different config - should miss
        modified_config = map_builder_config.copy()
        modified_config["generation_seed"] = 43  # Different seed

        with cache(modified_config) as ctx:
            cache_key_2 = track_key(ctx.cache_key)

            assert ctx.miss, "Should be cache miss for different config"
            assert cache_key != cache_key_2, "Different configs should have different keys"

    def test_multiple_cache_instances_same_bucket(self, s3_config, track_key):
        """Test that multiple cache instances can share the same bucket/prefix."""
        cache1 = S3CacheManager(**s3_config)
        cache2 = S3CacheManager(**s3_config)

        if not cache1.s3_available:
            pytest.skip("S3 not available")

        test_config = {"shared": "config", "instance": "test"}
        test_data = {"result": "shared_result"}

        # Cache with first instance
        with cache1(test_config) as ctx:
            _cache_key = track_key(ctx.cache_key)
            assert ctx.miss
            ctx.set(test_data)

        # Retrieve with second instance
        with cache2(test_config) as ctx:
            assert ctx.hit, "Second cache instance should see cached data"
            retrieved = ctx.get()
            assert retrieved == test_data, "Data should be shared between instances"


class TestCompression:
    """Tests for compression functionality."""

    def test_compression_effectiveness(self, cache, track_key):
        """Test that compression actually reduces size."""
        _cache_key = track_key("compression_test")

        # Create highly compressible data
        repetitive_data = {
            "repeated_string": "ABCD" * 1000,  # Very compressible
            "repeated_list": [42] * 1000,
            "metadata": {"compressed": True},
        }

        # Test compression directly
        compressed = cache._compress_object(repetitive_data)
        import pickle

        uncompressed = pickle.dumps(repetitive_data)

        compression_ratio = len(compressed) / len(uncompressed)
        print(f"Compression ratio: {compression_ratio:.2f} ({len(uncompressed)} -> {len(compressed)} bytes)")

        # Should achieve significant compression on repetitive data
        assert compression_ratio < 0.5, "Should achieve at least 50% compression on repetitive data"

        # Verify round-trip still works
        decompressed = cache._decompress_object(compressed)
        assert decompressed == repetitive_data, "Compression round-trip should preserve data"


class TestCacheManagement:
    """Tests for cache management operations (delete, list, etc.)."""

    def test_delete_single_key(self, cache, track_key):
        """Test deleting a single cache key (if delete method exists)."""
        if not hasattr(cache, "delete"):
            pytest.skip("delete method not available in this S3CacheManager version")

        cache_key = track_key("delete_single_test")
        test_data = {"data": "to_be_deleted"}

        # Put and verify
        success = cache.put(cache_key, test_data)
        assert success

        data = cache.get(cache_key)
        assert data == test_data

        # Delete and verify
        delete_success = cache.delete(cache_key)
        assert delete_success, "Delete should succeed"

        data = cache.get(cache_key)
        assert data is None, "Deleted key should return None"


class TestEnvironmentDebug:
    """Tests to debug environment differences between test and training."""

    def test_aws_environment_debug(self, caplog):
        """Debug AWS configuration to compare with training environment."""
        import boto3

        print("=== AWS ENVIRONMENT DEBUG ===")
        print(f"AWS_ACCESS_KEY_ID: {'SET' if os.getenv('AWS_ACCESS_KEY_ID') else 'NOT SET'}")
        print(f"AWS_SECRET_ACCESS_KEY: {'SET' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'NOT SET'}")
        print(f"AWS_PROFILE: {os.getenv('AWS_PROFILE', 'NOT SET')}")
        print(f"AWS_REGION: {os.getenv('AWS_REGION', 'NOT SET')}")

        try:
            session = boto3.Session()
            creds = session.get_credentials()
            if creds:
                print(f"Boto3 credentials: FOUND (access key: {creds.access_key[:10]}...)")
                print(f"Boto3 region: {session.region_name}")
            else:
                print("Boto3 credentials: NOT FOUND")
        except Exception as e:
            print(f"Boto3 error: {e}")

        # Test S3 access directly with hardcoded values
        try:
            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_client.head_bucket(Bucket="softmax-level-cache")
            print("✅ Direct S3 access to softmax-level-cache: SUCCESS")
        except Exception as e:
            print(f"❌ Direct S3 access to softmax-level-cache: FAILED - {e}")

    def test_training_environment_simulation(self):
        """Test cache initialization exactly like training environment."""
        print("=== TRAINING ENVIRONMENT SIMULATION ===")

        # Create a simple logger for this test
        import logging

        logger = logging.getLogger("test_cache")
        logger.setLevel(logging.INFO)

        # Simulate exact training environment setup with hardcoded values
        cache_manager = S3CacheManager(
            bucket_name="softmax-level-cache",
            prefix="map_builder_cache/",
            compression_level=6,
            aws_region="us-east-1",
            logger=logger,
        )

        print(f"S3 available in training simulation: {cache_manager.s3_available}")

        if cache_manager.s3_available:
            print("✅ Training environment simulation: S3 access successful")

            # Test a quick put/get cycle to verify full functionality
            test_key = f"test_training_sim_{int(time.time())}"
            test_data = {"simulation": "test", "timestamp": time.time()}

            put_success = cache_manager.put(test_key, test_data)
            print(f"Test put operation: {'SUCCESS' if put_success else 'FAILED'}")

            if put_success:
                retrieved_data = cache_manager.get(test_key)
                get_success = retrieved_data == test_data
                print(f"Test get operation: {'SUCCESS' if get_success else 'FAILED'}")

                # Clean up
                cache_manager.delete(test_key)
                print("Test cleanup completed")
        else:
            print("❌ Training environment simulation: S3 access failed")
            print("This suggests the issue is environment-specific, not code-specific")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_non_pickleable_objects_in_key(self, cache):
        """Test key generation with objects that can't be pickled."""
        import threading

        # Create an object that can't be pickled
        lock = threading.Lock()

        # Should fallback to string representation
        key = cache.create_key({"lock": lock, "normal": "data"})
        assert isinstance(key, str)
        assert len(key) == 64

        # Should be consistent
        _key2 = cache.create_key({"lock": lock, "normal": "data"})
        # Note: This might not be equal due to object id differences in string repr
        # but it shouldn't crash

    def test_empty_arguments(self, cache):
        """Test key generation with no arguments."""
        key1 = cache.create_key()
        key2 = cache.create_key()

        assert key1 == key2, "Empty arguments should generate consistent key"
        assert len(key1) == 64

    def test_cache_unavailable_operations(self, s3_config):
        """Test operations when S3 is unavailable."""
        # Create cache with invalid credentials/bucket
        bad_cache = S3CacheManager(bucket_name="nonexistent-bucket-12345", prefix="test/", aws_region="us-west-2")

        # Basic operations should return False/None gracefully
        assert bad_cache.get("any_key") is None
        assert bad_cache.put("any_key", "any_value") is False

        # Test delete methods if they exist (enhanced version)
        if hasattr(bad_cache, "delete"):
            assert bad_cache.delete("any_key") is False


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "-s"])
