import sys
import weakref
from typing import Any, Set


def get_object_size(obj: Any, visited: Set[int] = None) -> int:
    """Get the deep memory usage of an object in bytes, handling circular references."""
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return 0

    visited.add(obj_id)

    try:
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += get_object_size(key, visited)
                size += get_object_size(value, visited)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += get_object_size(item, visited)

        if hasattr(obj, "__dict__") and obj.__dict__:
            size += get_object_size(obj.__dict__, visited)

        if hasattr(obj, "__slots__"):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    size += get_object_size(getattr(obj, slot), visited)

    except (TypeError, RecursionError, AttributeError):
        pass

    finally:
        visited.discard(obj_id)

    return size


class MemoryMonitor:
    """Monitor memory usage of tracked objects using weak references."""

    def __init__(self):
        self._tracked_objects: dict[str, dict[str, Any]] = {}

    def add(self, obj: Any, name: str | None = None) -> None:
        """Add object and its attributes to monitor using weak references."""
        if name is None:
            name = self._generate_name(obj)

        try:
            # Track the main object
            initial_size = get_object_size(obj)

            try:
                weak_ref = weakref.ref(obj, lambda ref, key=name: self._tracked_objects.pop(key, None))
                self._tracked_objects[name] = {
                    "object_ref": weak_ref,
                    "initial_size": initial_size,
                    "is_weak": True,
                }
            except TypeError:
                # Store direct reference for objects that don't support weak references
                self._tracked_objects[name] = {
                    "object_ref": obj,
                    "initial_size": initial_size,
                    "is_weak": False,
                }

            # Track each attribute separately
            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    attr_key = f"{name}.{attr_name}"
                    try:
                        attr_initial_size = get_object_size(attr_value)
                        try:
                            attr_weak_ref = weakref.ref(
                                attr_value, lambda ref, key=attr_key: self._tracked_objects.pop(key, None)
                            )
                            self._tracked_objects[attr_key] = {
                                "object_ref": attr_weak_ref,
                                "initial_size": attr_initial_size,
                                "is_weak": True,
                            }
                        except TypeError:
                            # Store direct reference for attributes that don't support weak references
                            self._tracked_objects[attr_key] = {
                                "object_ref": attr_value,
                                "initial_size": attr_initial_size,
                                "is_weak": False,
                            }
                    except Exception:
                        # Skip attributes that can't be measured
                        pass

        except Exception as e:
            print(f"Warning: Could not track object {name}: {e}")

    def _generate_name(self, obj: Any) -> str:
        """Generate a name for an object based on its type and id."""
        obj_type = type(obj).__name__
        obj_id = str(id(obj))[-6:]

        if hasattr(obj, "__name__"):
            return f"{obj.__name__}_{obj_type}"
        elif hasattr(obj, "name") and isinstance(obj.name, str):
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
        if name not in self._tracked_objects:
            return {}

        info = self._tracked_objects[name]

        # Get object - either from weak ref or direct ref
        if info["is_weak"]:
            obj = info["object_ref"]()
            if obj is None:
                return {"status": "garbage_collected"}
        else:
            obj = info["object_ref"]

        try:
            current_size = get_object_size(obj)
            initial_size = info["initial_size"]
            size_change = current_size - initial_size

            return {
                "current_size": current_size,
                "current_size_mb": round(current_size / (1024 * 1024), 3),
                "initial_size": initial_size,
                "size_change": size_change,
                "size_change_mb": round(size_change / (1024 * 1024), 3),
                "is_weak": info["is_weak"],
            }
        except Exception:
            return {"status": "error"}

    def stats(self) -> dict[str, float]:
        """Get statistics for all tracked objects."""
        if not self._tracked_objects:
            return {}

        object_stats = {}
        for name in self._tracked_objects.keys():
            obj_info = self.get(name)
            if "current_size_mb" in obj_info and obj_info["current_size_mb"] > 0.001:
                object_stats[name] = obj_info["current_size_mb"]

        return object_stats
