"""Simple LRU cache for checkpoint loading - addresses performance issues from Phase 11 audit."""

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CheckpointCache:
    """Thread-safe LRU cache for checkpoint data.

    This addresses the performance regression identified in the audit where
    removing the PolicyCache caused 10-100x slower repeated access patterns.
    """

    def __init__(self, max_size: int = 10):
        """Initialize cache with maximum size."""
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to end (most recently used)."""
        with self._lock:
            if key not in self._cache:
                return None

            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting oldest if necessary."""
        with self._lock:
            if key in self._cache:
                # Update existing item and move to end
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new item
                self._cache[key] = value

                # Evict oldest if over capacity
                while len(self._cache) > self._max_size:
                    oldest_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"Evicted cached checkpoint: {oldest_key}")

    def remove(self, key: str) -> bool:
        """Remove specific item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            logger.debug("Cleared checkpoint cache")

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    def hit_ratio(self) -> float:
        """Get cache hit ratio (requires hit/miss tracking to be enabled)."""
        # For now, return 0.0 - hit/miss tracking can be added later if needed
        return 0.0

    def keys(self) -> list[str]:
        """Get list of cached keys (for debugging)."""
        with self._lock:
            return list(self._cache.keys())
