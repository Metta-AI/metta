from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import random
import tempfile

os.environ.setdefault("MP_NO_RESOURCE_TRACKER", "1")
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilderConfig

"""
This is an LLM Generated memory cache for GameMap objects across processes. Ideally
we make MapGen fast enough to not need this, but for now it allows training on
large maps.
"""

logger = logging.getLogger(__name__)

# Global cache registry file and lock - these coordinate access across processes
_registry_file: Optional[Path] = None
_lock_file: Optional[Path] = None
_maps_per_key: Optional[int] = None


def _get_registry_file() -> Path:
    """Get the registry file path for storing cache metadata across processes."""
    global _registry_file
    if _registry_file is None:
        # Use a temporary directory that's accessible across processes
        # On Linux, /dev/shm is typically available and is a tmpfs
        # Fall back to system temp directory if /dev/shm doesn't exist
        if os.path.exists("/dev/shm"):
            temp_base = Path("/dev/shm")
        else:
            temp_base = Path(tempfile.gettempdir())
        _registry_file = temp_base / "mettagrid_map_cache_registry.json"
    return _registry_file


def _get_lock_file() -> Path:
    """Get the lock file path for coordinating cache access across processes."""
    global _lock_file
    if _lock_file is None:
        # Use same base directory as registry file
        registry_file = _get_registry_file()
        _lock_file = registry_file.parent / "mettagrid_map_cache.lock"
    return _lock_file


def _load_registry() -> dict[str, list[dict[str, Any]]]:
    """Load the cache registry from disk."""
    registry_file = _get_registry_file()
    if not registry_file.exists():
        return {}
    try:
        with open(registry_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_registry(registry: dict[str, list[dict[str, Any]]]) -> None:
    """Save the cache registry to disk."""
    registry_file = _get_registry_file()
    # Write atomically by writing to temp file then renaming
    temp_file = registry_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(registry, f)
        temp_file.replace(registry_file)
    except Exception as e:
        logger.warning(f"Error saving cache registry: {e}")
        if temp_file.exists():
            temp_file.unlink()


class SharedMapCache:
    """Shared memory cache for GameMap objects across processes."""

    def __init__(self, maps_per_key: Optional[int] = None):
        """Initialize the shared memory map cache.

        Args:
            maps_per_key: Number of maps to cache per key. None for unlimited.
        """
        global _maps_per_key
        if maps_per_key is not None and _maps_per_key is None:
            _maps_per_key = maps_per_key
        self._shm_registry: dict[str, shared_memory.SharedMemory] = {}

    def start(self) -> None:
        """Start the shared cache (no-op, registry is file-based)."""
        _get_registry_file()
        logger.debug("Started shared memory map cache")

    def stop(self) -> None:
        """Stop the manager process and clean up shared memory."""
        # Clean up all shared memory blocks
        for shm_name in list(self._shm_registry.keys()):
            try:
                shm = self._shm_registry.pop(shm_name)
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory {shm_name}: {e}")

        logger.debug("Stopped shared memory map cache manager")

    def _make_key(self, map_builder: MapBuilderConfig, num_agents: int) -> str:
        """Create a cache key from map builder config and num_agents."""
        # Hash the map builder config JSON and num_agents
        config_json = map_builder.model_dump_json()
        key_data = f"{config_json}:{num_agents}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get_or_create(self, map_builder: MapBuilderConfig, num_agents: int) -> GameMap:
        """Get an existing map from cache or create a new one.

        Args:
            map_builder: MapBuilderConfig to use for building the map.
            num_agents: Number of agents for the map.

        Returns:
            GameMap from cache or newly created.
        """
        cache_key = self._make_key(map_builder, num_agents)
        lock_file = _get_lock_file()

        def _reconstruct_or_refresh(cache_entry: dict) -> GameMap:
            """Rebuild map when cached shared memory is missing."""
            try:
                return self._reconstruct_map(cache_entry, cache_key)
            except (FileNotFoundError, PermissionError, OSError):
                registry = _load_registry()
                maps_list = registry.get(cache_key, [])
                builder = map_builder.create()
                game_map = builder.build_for_num_agents(num_agents)
                cache_entry = self._store_map_in_shared_memory(cache_key, len(maps_list), game_map)
                registry[cache_key] = list(maps_list) + [cache_entry]
                _save_registry(registry)
                logger.info(f"Rebuilt stale map for key {cache_key}")
                return game_map

        # Use file-based locking to coordinate access across processes
        # This ensures that spawned processes can coordinate cache access
        try:
            with open(lock_file, "a+") as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                try:
                    # Load registry from disk
                    registry = _load_registry()

                    # Get or create the cache entry for this key
                    if cache_key not in registry:
                        registry[cache_key] = []

                    maps_list = registry[cache_key]

                    # Decide whether to create new map or reuse cached one
                    should_create_new = False
                    if _maps_per_key is None:
                        # Unlimited: create new only if cache is empty
                        should_create_new = len(maps_list) == 0
                    else:
                        # Limited: create new only if under limit
                        should_create_new = len(maps_list) < _maps_per_key

                    if should_create_new:
                        # Create a new map
                        builder = map_builder.create()
                        game_map = builder.build_for_num_agents(num_agents)

                        # Reload registry before saving to check if another process created it
                        registry = _load_registry()
                        if cache_key not in registry:
                            registry[cache_key] = []
                        maps_list = registry[cache_key]

                        # Check again if we still need to create (another process might have created it)
                        if _maps_per_key is None:
                            still_need_create = len(maps_list) == 0
                        else:
                            still_need_create = len(maps_list) < _maps_per_key

                        if still_need_create:
                            # Store it in cache
                            cache_entry = self._store_map_in_shared_memory(cache_key, len(maps_list), game_map)
                            maps_list = list(maps_list) + [cache_entry]
                            registry[cache_key] = maps_list
                            _save_registry(registry)

                            logger.info(
                                f"Created new map for key {cache_key} "
                                f"(cached maps: {len(maps_list)}/{_maps_per_key if _maps_per_key else 'unlimited'})"
                            )
                            return game_map
                        else:
                            # Another process created it, use the cached one instead
                            random_idx = random.randint(0, len(maps_list) - 1)
                            cache_entry = maps_list[random_idx]
                            return _reconstruct_or_refresh(cache_entry)
                    else:
                        # Array is full, return a random one
                        random_idx = random.randint(0, len(maps_list) - 1)
                        cache_entry = maps_list[random_idx]

                        return _reconstruct_or_refresh(cache_entry)
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            # If file locking fails (e.g., on Windows), fall back to unlocked access
            logger.debug(f"File locking not available ({e}), proceeding without lock")
            registry = _load_registry()
            if cache_key not in registry:
                registry[cache_key] = []
            maps_list = registry[cache_key]

            if _maps_per_key is None:
                should_create_new = len(maps_list) == 0
            else:
                should_create_new = len(maps_list) < _maps_per_key

            if should_create_new:
                builder = map_builder.create()
                game_map = builder.build_for_num_agents(num_agents)

                # Reload registry before saving to check if another process created it
                registry = _load_registry()
                if cache_key not in registry:
                    registry[cache_key] = []
                maps_list = registry[cache_key]

                # Check again if we still need to create (another process might have created it)
                if _maps_per_key is None:
                    still_need_create = len(maps_list) == 0
                else:
                    still_need_create = len(maps_list) < _maps_per_key

                if still_need_create:
                    cache_entry = self._store_map_in_shared_memory(cache_key, len(maps_list), game_map)
                    maps_list = list(maps_list) + [cache_entry]
                    registry[cache_key] = maps_list
                    _save_registry(registry)
                    logger.info(
                        f"Created new map for key {cache_key} "
                        f"(cached maps: {len(maps_list)}/{_maps_per_key if _maps_per_key else 'unlimited'})"
                    )
                    return game_map
                else:
                    # Another process created it, use the cached one instead
                    random_idx = random.randint(0, len(maps_list) - 1)
                    cache_entry = maps_list[random_idx]
                    return _reconstruct_or_refresh(cache_entry)
            else:
                random_idx = random.randint(0, len(maps_list) - 1)
                cache_entry = maps_list[random_idx]
                return _reconstruct_or_refresh(cache_entry)

    def _store_map_in_shared_memory(self, cache_key: str, index: int, game_map: GameMap) -> dict:
        """Store a GameMap in shared memory and return cache entry."""
        # Truncate cache_key for shared memory name (macOS POSIX limit is 31 chars)
        # Using first 16 chars of hash should be sufficient for uniqueness in practice
        short_key = cache_key[:16]
        # Format: "m{hash16}_{index}" - max length: 1 + 16 + 1 + digits(index) <= 31
        # For index up to 999: 1 + 16 + 1 + 3 = 21 chars (safe)
        shm_name = f"m{short_key}_{index}"
        grid = game_map.grid
        shape = grid.shape
        dtype_str = str(grid.dtype)
        nbytes = grid.nbytes

        # Clean up existing shared memory if it exists in our registry
        if shm_name in self._shm_registry:
            try:
                old_shm = self._shm_registry.pop(shm_name)
                old_shm.close()
                old_shm.unlink()
            except Exception:
                pass

        # Create new shared memory block
        # Handle case where shared memory already exists (e.g., from previous test run or process)
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
            self._shm_registry[shm_name] = shm
        except FileExistsError:
            # Shared memory exists but not in our registry - try to unlink it
            try:
                existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                existing_shm.close()
                existing_shm.unlink()
                logger.debug(f"Unlinked existing shared memory {shm_name} before recreating")
                # Retry creating after unlinking
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
                self._shm_registry[shm_name] = shm
            except Exception:
                # If unlink fails, try to use the existing shared memory (might be valid)
                logger.warning(f"Could not unlink existing shared memory {shm_name}, attempting to reuse")
                try:
                    existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                    # Check if size matches
                    if existing_shm.size == nbytes:
                        shm = existing_shm
                        self._shm_registry[shm_name] = shm
                        logger.debug(f"Reusing existing shared memory {shm_name}")
                    else:
                        # Size mismatch - force unlink and recreate
                        existing_shm.close()
                        existing_shm.unlink()
                        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
                        self._shm_registry[shm_name] = shm
                        logger.debug(f"Recreated shared memory {shm_name} due to size mismatch")
                except Exception as cleanup_err:
                    # Last resort: raise the original FileExistsError
                    raise FileExistsError(
                        f"Shared memory {shm_name} exists and could not be cleaned up. "
                        f"This may indicate a stale shared memory block from a previous process."
                    ) from cleanup_err

        # Copy grid data to shared memory
        shm_grid = np.ndarray(shape, dtype=grid.dtype, buffer=shm.buf)
        np.copyto(shm_grid, grid)

        # Return cache entry metadata
        return {
            "shm_name": shm_name,
            "shape": list(shape),  # Convert to list for serialization
            "dtype": dtype_str,
        }

    def _reconstruct_map(self, cache_entry: dict, cache_key: str) -> GameMap:
        """Reconstruct a GameMap from cache entry."""
        shm_name = cache_entry["shm_name"]
        shape = tuple(cache_entry["shape"])
        dtype_str = cache_entry["dtype"]

        # Get or create shared memory handle
        if shm_name not in self._shm_registry:
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
                self._shm_registry[shm_name] = shm
            except (FileNotFoundError, PermissionError, OSError):
                # Shared memory errors should crash the process - don't catch and retry
                # Clean up registry entry first, then re-raise
                registry = _load_registry()
                maps_list = registry.get(cache_key)
                if maps_list is not None:
                    # Remove this entry from the list
                    maps_list = [e for e in maps_list if e.get("shm_name") != shm_name]
                    if not maps_list:
                        registry.pop(cache_key, None)
                    else:
                        registry[cache_key] = maps_list
                    _save_registry(registry)
                raise

        shm = self._shm_registry[shm_name]
        # Reconstruct numpy array from shared memory
        grid = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf).copy()
        return GameMap(grid=grid)

    def _remove_key_entry(self, cache_key: str) -> None:
        """Remove an entire key entry and all its cached maps."""
        registry = _load_registry()
        maps_list = registry.pop(cache_key, None)
        if maps_list is None:
            return

        # Clean up all map entries
        for cache_entry in maps_list:
            shm_name = cache_entry.get("shm_name") if isinstance(cache_entry, dict) else None
            if shm_name and shm_name in self._shm_registry:
                try:
                    shm = self._shm_registry.pop(shm_name)
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.debug(f"Error removing shared memory {shm_name}: {e}")

        _save_registry(registry)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        registry = _load_registry()
        # Remove all entries
        cache_keys = list(registry.keys())
        for cache_key in cache_keys:
            self._remove_key_entry(cache_key)

        logger.debug("Cleared shared memory map cache")

    def __len__(self) -> int:
        """Get the total number of cached maps across all keys."""
        registry = _load_registry()
        total = 0
        for maps_list in registry.values():
            total += len(maps_list)
        return total


# Global shared cache instance
_shared_cache: Optional[SharedMapCache] = None


def get_shared_cache(maps_per_key: Optional[int] = None) -> SharedMapCache:
    """Get or create the global shared memory map cache.

    Args:
        maps_per_key: Number of maps to cache per key. Only used on first call.

    Returns:
        The global SharedMapCache instance.
    """
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = SharedMapCache(maps_per_key=maps_per_key)
        _shared_cache.start()
    return _shared_cache


def stop_shared_cache() -> None:
    """Stop the global shared memory map cache."""
    global _shared_cache
    if _shared_cache is not None:
        _shared_cache.stop()
        _shared_cache = None
