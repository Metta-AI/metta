import inspect
import sys
import weakref
from typing import Any, Set


def get_object_size(obj: Any, visited: Set[int] | None = None) -> int:
    """Get the deep memory usage of an object in bytes, handling circular references."""
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return 0

    visited.add(obj_id)
    size = 0

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

        slots = getattr(obj, "__slots__", None)
        if slots is not None:
            for slot in slots:
                if hasattr(obj, slot):
                    try:
                        slot_value = getattr(obj, slot)
                        size += get_object_size(slot_value, visited)
                    except (AttributeError, ValueError):
                        # Skip slots that can't be accessed
                        pass

    except (TypeError, RecursionError, AttributeError):
        pass
    except Exception:
        # Catch any other unexpected exceptions
        pass

    return size


class MemoryMonitor:
    """Monitor memory usage of tracked objects using weak references."""

    def __init__(self):
        self._tracked_objects: dict[str, dict[str, Any]] = {}

    def add(self, obj: Any, name: str | None = None, track_attributes: bool = False) -> None:
        """Add object to monitor using weak references.

        Args:
            obj: The object to track
            name: Optional name for the object. If None, auto-generated.
            track_attributes: If True, also track individual attributes separately
        """
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

            # Only track attributes if requested
            if track_attributes:
                self._track_object_fields(obj, name)

        except Exception as e:
            print(f"Warning: Could not track object {name}: {e}")

    def _track_object_fields(self, obj: Any, name: str) -> None:
        """Track each attribute/field of an object separately."""
        # Track each __dict__ attribute separately
        if hasattr(obj, "__dict__") and obj.__dict__:
            for attr_name, attr_value in obj.__dict__.items():
                attr_key = f"{name}.{attr_name}"
                try:
                    attr_initial_size = get_object_size(attr_value)
                    try:
                        attr_weak_ref = weakref.ref(
                            attr_value, lambda _ref, key=attr_key: self._tracked_objects.pop(key, None)
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

        # Handle __slots__ attributes separately
        slots = getattr(obj, "__slots__", None)
        if slots is not None:
            for slot in slots:
                if hasattr(obj, slot):
                    try:
                        slot_value = getattr(obj, slot)
                        attr_key = f"{name}.{slot}"
                        attr_initial_size = get_object_size(slot_value)

                        try:
                            attr_weak_ref = weakref.ref(
                                slot_value, lambda _ref, key=attr_key: self._tracked_objects.pop(key, None)
                            )
                            self._tracked_objects[attr_key] = {
                                "object_ref": attr_weak_ref,
                                "initial_size": attr_initial_size,
                                "is_weak": True,
                            }
                        except TypeError:
                            # Store direct reference for slot values that don't support weak references
                            self._tracked_objects[attr_key] = {
                                "object_ref": slot_value,
                                "initial_size": attr_initial_size,
                                "is_weak": False,
                            }
                    except (AttributeError, ValueError, Exception):
                        # Skip slots that can't be accessed or measured
                        pass

    def _generate_name(self, obj: Any) -> str:
        """Generate a name for an object based on its type and caller location."""
        obj_type = type(obj).__name__

        # Get caller information (the code that called add())
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                filename = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno

                # Extract just the filename without path
                short_filename = filename.split("/")[-1].split("\\")[-1]
                location = f"{short_filename}:{line_number}"
            else:
                location = "unknown"
        except Exception:
            location = "unknown"

        if hasattr(obj, "__name__"):
            return f"{location}.{obj.__name__}[{obj_type}]"
        elif hasattr(obj, "name") and isinstance(obj.name, str):
            return f"{location}.{obj.name}[{obj_type}]"
        else:
            return f"{location}.{obj_type}"

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
