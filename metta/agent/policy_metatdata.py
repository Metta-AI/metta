import logging
from typing import Any, Optional


class PolicyMetadata(dict[str, Any]):
    """Dict-like metadata with required fields and support for additional arbitrary fields."""

    # Define required field names as class constant
    _REQUIRED_FIELDS = {"agent_step", "epoch", "generation", "train_time"}

    def __init__(self, agent_step: int, epoch: int, generation: int, train_time: float, **kwargs: Any):
        """Initialize with required fields and optional additional fields."""
        # Initialize the dict with all fields
        super().__init__(agent_step=agent_step, epoch=epoch, generation=generation, train_time=train_time, **kwargs)

    def __delitem__(self, key: str) -> None:
        """Delete item by key, preventing deletion of required fields."""
        if key in self._REQUIRED_FIELDS:
            raise KeyError(f"Cannot delete required field: {key}")
        super().__delitem__(key)

    def pop(self, key: str, default: Any = ...) -> Any:
        """Remove and return value for key, preventing removal of required fields."""
        if key in self._REQUIRED_FIELDS:
            raise KeyError(f"Cannot pop required field: {key}")

        if default is ...:
            return super().pop(key)
        else:
            return super().pop(key, default)

    def sanitized(self, logger: Optional[logging.Logger] = None) -> "PolicyMetadata":
        """Return a sanitized copy safe for pickling.

        Removes wandb-related objects and converts non-picklable objects to strings.

        Args:
            logger: Optional logger for warnings about sanitized values
        """

        def sanitize_value(val: Any, path: str = "") -> Any:
            # Skip wandb-related objects
            if hasattr(val, "__module__") and val.__module__ and "wandb" in val.__module__:
                if logger:
                    logger.warning(
                        f"Removing wandb object at {path or 'root'}: {type(val).__name__} from module {val.__module__}"
                    )
                return None

            # Recursively clean dictionaries
            if isinstance(val, dict):
                sanitized = {}
                for k, v in val.items():
                    new_path = f"{path}.{k}" if path else str(k)
                    sanitized_val = sanitize_value(v, new_path)
                    if sanitized_val is not None:
                        sanitized[k] = sanitized_val
                    elif logger and v is not None:
                        # Log if we're removing a non-None value
                        logger.warning(f"Removed non-picklable value at {new_path}")
                return sanitized

            # Recursively clean lists
            if isinstance(val, list):
                sanitized = []
                for i, item in enumerate(val):
                    new_path = f"{path}[{i}]"
                    sanitized_val = sanitize_value(item, new_path)
                    if sanitized_val is not None:
                        sanitized.append(sanitized_val)
                    elif logger and item is not None:
                        logger.warning(f"Removed non-picklable value at {new_path}")
                return sanitized

            # Keep primitive types as-is
            if isinstance(val, (str, int, float, bool, type(None))):
                return val

            # Convert objects to strings, return None if conversion fails
            if hasattr(val, "__dict__"):
                try:
                    str_val = str(val)
                    if logger:
                        logger.warning(
                            f"Converted object to string at {path or 'root'}: {type(val).__name__} -> '{str_val}'"
                        )
                    return str_val
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to convert object at {path or 'root'}: {type(val).__name__} - {str(e)}")
                    return None

            # Keep everything else as-is
            return val

        # Sanitize all values
        sanitized_dict = {}
        for key, value in self.items():
            sanitized_val = sanitize_value(value, key)
            if sanitized_val is not None:
                sanitized_dict[key] = sanitized_val
            elif key in self._REQUIRED_FIELDS:
                # Required fields should always have a value
                if logger:
                    logger.warning(f"Required field '{key}' was sanitized to None")
                sanitized_dict[key] = None

        # Create new instance with sanitized values
        return PolicyMetadata(**sanitized_dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyMetadata":
        """Create PolicyMetadata from a dictionary, validating required fields."""
        # Check for required fields
        missing_fields = cls._REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return cls(**data)

    def __repr__(self) -> str:
        """String representation showing all fields."""
        items = []
        # Show required fields first
        for key in sorted(self._REQUIRED_FIELDS):
            items.append(f"{key}={self[key]!r}")
        # Then show other fields
        for key, value in sorted(self.items()):
            if key not in self._REQUIRED_FIELDS:
                items.append(f"{key}={value!r}")
        return f"PolicyMetadata({', '.join(items)})"
