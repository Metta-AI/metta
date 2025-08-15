"""
LRU Cache implementation for PolicyRecord objects.
Manages in-memory caching of loaded policies with automatic eviction.
"""

import logging
from collections import OrderedDict
from threading import RLock
from typing import Optional

from metta.agent.policy_record import PolicyRecord

logger = logging.getLogger(__name__)


class PolicyCache:
    """
    Thread-safe LRU cache for PolicyRecord objects.

    Automatically evicts least recently used policies when the cache
    reaches its maximum size, preventing excessive memory usage.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize the policy cache.

        Args:
            max_size: Maximum number of policies to keep in cache.
                     Defaults to 10 to prevent memory issues.
        """
        if max_size <= 0:
            raise ValueError("Cache size must be positive")

        self._max_size = max_size
        self._cache: OrderedDict[str, PolicyRecord] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Optional[PolicyRecord]:
        """
        Retrieve a policy record from cache.

        Args:
            key: The URI or path used as cache key

        Returns:
            PolicyRecord if found in cache, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                logger.debug(f"Cache miss for key: {key}")
                return None

            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            logger.debug(f"Cache hit for key: {key}")
            return self._cache[key]

    def put(self, key: str, record: PolicyRecord) -> None:
        """
        Store a policy record in cache.

        If cache is at capacity, the least recently used item is evicted.

        Args:
            key: The URI or path to use as cache key
            record: The PolicyRecord to cache
        """
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = record
                self._cache.move_to_end(key)
                logger.debug(f"Updated existing cache entry: {key}")
                return

            # Add new entry
            self._cache[key] = record

            # Evict LRU if over capacity
            if len(self._cache) > self._max_size:
                evicted_key, evicted_record = self._cache.popitem(last=False)
                logger.info(
                    f"Evicted policy from cache: {evicted_record.run_name} "
                    f"(key: {evicted_key}) - Cache at capacity ({self._max_size})"
                )
