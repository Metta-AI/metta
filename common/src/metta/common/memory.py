import sys
from typing import Any


def get_object_size(obj: Any) -> int:
    """Get the deep memory usage of an object in bytes."""
    # For simple objects, use sys.getsizeof
    size = sys.getsizeof(obj)

    # For containers, try to get deep size
    if isinstance(obj, dict):
        size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(sys.getsizeof(item) for item in obj)

    # For objects with __dict__, include attribute sizes
    if hasattr(obj, "__dict__"):
        size += sys.getsizeof(obj.__dict__)
        size += sum(get_object_size(v) for v in obj.__dict__.values())

    return size
