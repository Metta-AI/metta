"""Pytest collection hook for deterministic sharding."""

from __future__ import annotations

import hashlib
import os

from _pytest.config import Config
from _pytest.nodes import Item


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Filter collected *items* so only the current shard executes.

    Environment variables ``PYTEST_SHARD_ID`` and ``PYTEST_NUM_SHARDS`` control
    the sharding behaviour. When ``PYTEST_NUM_SHARDS`` is unset or ``1`` the
    collection remains untouched.
    """

    shard_id = int(os.environ.get("PYTEST_SHARD_ID", "0"))
    num_shards = int(os.environ.get("PYTEST_NUM_SHARDS", "1"))

    if num_shards <= 1:
        return

    items.sort(key=lambda item: item.nodeid)

    selected: list[Item] = []
    deselected: list[Item] = []

    for item in items:
        item_hash = int(hashlib.md5(item.nodeid.encode(), usedforsecurity=False).hexdigest(), 16)
        if item_hash % num_shards == shard_id:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)

    print(f"Shard {shard_id}/{num_shards}: running {len(selected)} of {len(selected) + len(deselected)} tests")
