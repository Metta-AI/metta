# Metta Utilities

This directory contains utility functions and classes for the Metta project.

## Distributed Locking (dist.py)

The `dist.py` module provides utilities for distributed locking, particularly useful when running multiple jobs that share the same AWS EFS mount.

### EfsLock

`EfsLock` is a class that implements a distributed lock using file-based locking with timestamp-based stale lock detection.

```python
from util.dist import EfsLock

# Using the class directly
with EfsLock("/path/to/lock", timeout=300):
    # This code will only run in one process at a time
    do_something()
```

### efs_lock

`efs_lock` is a convenience function that wraps the `EfsLock` class.

```python
from util.dist import efs_lock

# Using the convenience function
with efs_lock("/path/to/lock", timeout=300):
    # This code will only run in one process at a time
    do_something()
```

### Parameters

Both `EfsLock` and `efs_lock` accept the following parameters:

- `path`: Path to the lock file
- `timeout`: Time in seconds after which a lock is considered stale (default: 300)
- `retry_interval`: Time in seconds to wait between retries (default: 5)
- `max_retries`: Maximum number of times to retry acquiring the lock (default: 60)

### Example: Creating a Sweep

Here's an example of how to use `efs_lock` to ensure only one process creates a sweep:

```python
from util.dist import efs_lock
import os

def create_sweep(sweep_name, config):
    # Check if sweep already exists
    if os.path.exists(f"{config.sweep_dir}/config.yaml"):
        print(f"Sweep already exists: {sweep_name}")
        return

    # Create necessary directories
    os.makedirs(config.sweep_dir, exist_ok=True)

    # Use efs_lock to ensure only one process creates the sweep
    lock_file = os.path.join(config.sweep_dir, "lock")
    try:
        with efs_lock(lock_file, timeout=300):
            # Double-check that sweep wasn't created while waiting for lock
            if os.path.exists(f"{config.sweep_dir}/config.yaml"):
                print(f"Sweep was created while waiting for lock")
                return

            print(f"Creating new sweep: {sweep_name}")
            # ... create the sweep ...

    except TimeoutError:
        print(f"Timed out waiting for lock to create sweep: {sweep_name}")
```
