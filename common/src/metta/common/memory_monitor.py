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


class MemoryMonitor:
    """Monitor memory usage of tracked objects."""

    def __init__(self):
        self._tracked_objects: dict[str, dict[str, Any]] = {}

    def add(self, object: Any, name: str | None = None) -> None:
        """Add object and its attributes to monitor."""
        if name is None:
            name = self._generate_name(object)

        # Track the main object
        self._tracked_objects[name] = {
            "object": object,
            "initial_size": get_object_size(object),
        }

        # Track each attribute separately
        if hasattr(object, "__dict__"):
            for attr_name, attr_value in object.__dict__.items():
                attr_key = f"{name}.{attr_name}"
                self._tracked_objects[attr_key] = {
                    "object": attr_value,
                    "initial_size": get_object_size(attr_value),
                }

    def _generate_name(self, obj: Any) -> str:
        """Generate a name for an object based on its type and id."""
        obj_type = type(obj).__name__
        obj_id = str(id(obj))[-6:]  # Last 6 digits of memory address

        # Try to get a more descriptive name
        if hasattr(obj, "__name__"):
            return f"{obj.__name__}_{obj_type}"
        elif hasattr(obj, "name"):
            return f"{obj.name}_{obj_type}"
        else:
            return f"{obj_type}_{obj_id}"

    def remove(self, name: str) -> bool:
        """Remove an object and its attributes from monitoring."""
        removed = False
        # Remove main object
        if self._tracked_objects.pop(name, None) is not None:
            removed = True

        # Remove all attributes
        keys_to_remove = [key for key in self._tracked_objects.keys() if key.startswith(f"{name}.")]
        for key in keys_to_remove:
            self._tracked_objects.pop(key)
            removed = True

        return removed

    def clear(self) -> None:
        """Remove all tracked objects."""
        self._tracked_objects.clear()

    def get(self, name: str) -> dict[str, Any]:
        """Get stats for a specific tracked object."""
        if name in self._tracked_objects:
            info = self._tracked_objects[name]
            current_size = get_object_size(info["object"])
            initial_size = info["initial_size"]
            size_change = current_size - initial_size

            return {
                "current_size": current_size,
                "current_size_mb": round(current_size / (1024 * 1024), 3),
                "initial_size": initial_size,
                "size_change": size_change,
                "size_change_mb": round(size_change / (1024 * 1024), 3),
            }
        return {}

    def stats(self) -> dict[str, float]:
        """Get statistics for all tracked objects."""
        if not self._tracked_objects:
            return {}

        object_stats = {}
        for name in self._tracked_objects.keys():
            object_stats[f"{name}.size_mb"] = self.get(name)["current_size_mb"]

        return object_stats
