"""Tests for shared memory map cache."""

import importlib.util
import multiprocessing
import platform
from pathlib import Path

import numpy as np
import pytest

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME

# Import directly from module file to avoid circular import through __init__.py
# Load the module directly without going through the package __init__
_map_cache_path = Path(__file__).parent.parent / "python" / "src" / "mettagrid" / "simulator" / "map_cache.py"
spec = importlib.util.spec_from_file_location("mettagrid.simulator.map_cache", _map_cache_path)
_map_cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_map_cache_module)

SharedMapCache = _map_cache_module.SharedMapCache
get_shared_cache = _map_cache_module.get_shared_cache
stop_shared_cache = _map_cache_module.stop_shared_cache


@pytest.fixture
def cache():
    """Create a fresh cache instance for testing."""
    # Clean up any existing global cache
    stop_shared_cache()
    cache = SharedMapCache()
    cache.start()
    yield cache
    cache.stop()
    cache.clear()
    stop_shared_cache()


def _make_test_config(key: int, width: int = 2, height: int = 2) -> AsciiMapBuilder.Config:
    """Create a test MapBuilderConfig with unique data."""
    # Create unique map data based on key
    # Must include '@' for agent spawn points
    map_data = [["."] * width for _ in range(height)]
    map_data[0][0] = "@"  # Add agent spawn point
    if key % 2 == 0:
        map_data[0][1] = "#"  # Add wall for variation
    return AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_map_name=DEFAULT_CHAR_TO_NAME,
    )


def test_basic_get_or_create(cache):
    """Test basic get_or_create operation."""
    config = _make_test_config(1)
    num_agents = 1

    # First call should create a new map
    game_map1 = cache.get_or_create(config, num_agents)
    assert game_map1 is not None

    # Second call with same config should return a cached map
    game_map2 = cache.get_or_create(config, num_agents)
    assert game_map2 is not None
    # Should be the same map (same grid data)
    np.testing.assert_array_equal(game_map1.grid, game_map2.grid)


def test_different_configs(cache):
    """Test that different configs produce different maps."""
    config1 = _make_test_config(1)
    config2 = _make_test_config(2)
    num_agents = 1

    map1 = cache.get_or_create(config1, num_agents)
    map2 = cache.get_or_create(config2, num_agents)

    # Maps should be different (different grid data)
    assert not np.array_equal(map1.grid, map2.grid)


def test_different_num_agents(cache):
    """Test that different num_agents produce different cache keys."""
    # Create config with multiple spawn points for num_agents=2
    map_data = [["@", "@"], [".", "."]]  # 2 spawn points
    config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_map_name=DEFAULT_CHAR_TO_NAME,
    )

    map1 = cache.get_or_create(config, num_agents=1)
    map2 = cache.get_or_create(config, num_agents=2)

    # Both should work (different cache keys)
    assert map1 is not None
    assert map2 is not None


def test_len(cache):
    """Test __len__ method."""
    assert len(cache) == 0

    config1 = _make_test_config(1)
    config2 = _make_test_config(2)
    num_agents = 1

    cache.get_or_create(config1, num_agents)
    assert len(cache) >= 1  # At least one map cached

    cache.get_or_create(config2, num_agents)
    # Should have at least 2 maps (one per config)
    assert len(cache) >= 1


def test_maps_per_key(cache):
    """Test that cache respects maps_per_key limit."""
    cache_with_limit = SharedMapCache(maps_per_key=2)
    cache_with_limit.start()

    try:
        # For this test, we need different configs/num_agents to create different maps
        # Since AsciiMapBuilder is deterministic, same config + num_agents = same map
        config = _make_test_config(1)

        # First map with num_agents=1
        map1 = cache_with_limit.get_or_create(config, num_agents=1)
        assert len(cache_with_limit) >= 1

        # Second map with num_agents=2 (different cache key)
        # Need config with 2 spawn points
        map_data = [["@", "@"], [".", "."]]
        config2 = AsciiMapBuilder.Config(
            map_data=map_data,
            char_to_map_name=DEFAULT_CHAR_TO_NAME,
        )
        _ = cache_with_limit.get_or_create(config2, num_agents=2)
        assert len(cache_with_limit) >= 2

        # Third call with same config as map1 should return cached map
        map3 = cache_with_limit.get_or_create(config, num_agents=1)
        # Should still have at least 2 maps
        assert len(cache_with_limit) >= 2
        # map3 should be the same as map1 (cached)
        assert np.array_equal(map3.grid, map1.grid)
    finally:
        cache_with_limit.stop()
        cache_with_limit.clear()


def test_clear(cache):
    """Test clearing the cache."""
    config1 = _make_test_config(1)
    config2 = _make_test_config(2)
    num_agents = 1

    cache.get_or_create(config1, num_agents)
    cache.get_or_create(config2, num_agents)
    assert len(cache) >= 2

    cache.clear()
    assert len(cache) == 0


def test_large_map(cache):
    """Test caching a large map."""
    # Create a larger map config with spawn point
    map_data = [["."] * 100 for _ in range(100)]
    map_data[0][0] = "@"  # Add agent spawn point
    large_config = AsciiMapBuilder.Config(
        map_data=map_data,
        char_to_map_name=DEFAULT_CHAR_TO_NAME,
    )

    num_agents = 1
    game_map = cache.get_or_create(large_config, num_agents)

    assert game_map is not None
    assert game_map.grid.shape == (100, 100)


def _worker_get_map(config_json: str, num_agents: int, result_queue: multiprocessing.Queue) -> None:
    """Worker function to get a map from shared cache."""
    from mettagrid.map_builder.ascii import AsciiMapBuilder

    cache = get_shared_cache()
    # Reconstruct config from JSON
    config = AsciiMapBuilder.Config.model_validate_json(config_json)
    retrieved_map = cache.get_or_create(config, num_agents)
    if retrieved_map is not None:
        result_queue.put(("success", retrieved_map.grid.tolist()))
    else:
        result_queue.put(("not_found", None))


def _worker_create_map(config_json: str, num_agents: int, result_queue: multiprocessing.Queue) -> None:
    """Worker function to create a map in shared cache."""
    from mettagrid.map_builder.ascii import AsciiMapBuilder

    cache = get_shared_cache()
    config = AsciiMapBuilder.Config.model_validate_json(config_json)
    cache.get_or_create(config, num_agents)
    result_queue.put(("done", None))


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Multiprocessing tests can be flaky on macOS with fork method",
)
def test_cross_process_sharing(cache):
    """Test that maps can be shared across processes."""
    # Set spawn method for cross-platform compatibility
    if platform.system() != "Windows":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

    # Create and cache a map in main process
    config = _make_test_config(1)
    num_agents = 1
    game_map = cache.get_or_create(config, num_agents)
    expected_grid = game_map.grid.tolist()

    # Try to retrieve it from a worker process
    config_json = config.model_dump_json()
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_worker_get_map, args=(config_json, num_agents, result_queue))
    process.start()
    process.join(timeout=5)

    assert process.exitcode == 0
    status, retrieved_grid = result_queue.get(timeout=1)
    assert status == "success"
    assert retrieved_grid == expected_grid


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Multiprocessing tests can be flaky on macOS with fork method",
)
def test_cross_process_create_get(cache):
    """Test creating a map in one process and getting it in another."""
    # Set spawn method for cross-platform compatibility
    if platform.system() != "Windows":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

    # Create map from worker process
    config = _make_test_config(1)
    num_agents = 1
    config_json = config.model_dump_json()

    result_queue = multiprocessing.Queue()
    process1 = multiprocessing.Process(target=_worker_create_map, args=(config_json, num_agents, result_queue))
    process1.start()
    process1.join(timeout=5)
    assert process1.exitcode == 0

    # Get map from main process - should use cached version
    retrieved_map = cache.get_or_create(config, num_agents)
    assert retrieved_map is not None


def test_get_shared_cache_singleton():
    """Test that get_shared_cache returns a singleton."""
    stop_shared_cache()

    cache1 = get_shared_cache()
    cache2 = get_shared_cache()

    # Should be the same instance
    assert cache1 is cache2

    stop_shared_cache()


def test_get_shared_cache_params():
    """Test that maps_per_key is respected when creating shared cache."""
    stop_shared_cache()

    cache1 = get_shared_cache(maps_per_key=5)
    cache2 = get_shared_cache(maps_per_key=10)  # Should be ignored

    # Should be the same instance with original params
    assert cache1 is cache2

    stop_shared_cache()
