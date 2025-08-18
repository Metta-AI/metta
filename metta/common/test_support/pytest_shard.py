"""Pytest sharding plugin for distributing tests across multiple CI jobs."""

import hashlib
import os


def pytest_collection_modifyitems(config, items):
    """Distribute tests across shards based on test nodeid hash."""
    shard_id = int(os.environ.get("PYTEST_SHARD_ID", "0"))
    num_shards = int(os.environ.get("PYTEST_NUM_SHARDS", "1"))

    if num_shards <= 1:
        return

    # Sort items to ensure consistent ordering across runs
    items.sort(key=lambda x: x.nodeid)

    # Filter items based on hash of test nodeid
    selected = []
    deselected = []

    for item in items:
        # Use MD5 hash to distribute tests evenly
        test_hash = int(hashlib.md5(item.nodeid.encode()).hexdigest(), 16)
        if test_hash % num_shards == shard_id:
            selected.append(item)
        else:
            deselected.append(item)

    # Update items list and notify pytest of deselected items
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)

    print(f"Shard {shard_id}/{num_shards}: Running {len(selected)} of {len(selected) + len(deselected)} tests")
