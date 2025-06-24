import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        size += sum(sys.getsizeof(v) for v in obj.__dict__.values())

    return size


def log_object_memory(obj: Any, name: Optional[str] = None, level: int = logging.INFO) -> None:
    """Log the memory usage of a specific object."""
    obj_name = name or f"Object_{id(obj)}"
    size_bytes = get_object_size(obj)
    size_mb = size_bytes / (1024 * 1024)

    logger.log(level, f"Memory usage for '{obj_name}': {size_mb:.2f} MB ({size_bytes:,} bytes)")
