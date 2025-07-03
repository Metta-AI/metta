import logging
from typing import Any

logger = logging.getLogger(__name__)


class PolicyMetadata(dict[str, Any]):
    """Dict-like metadata with required fields and support for additional arbitrary fields."""

    # Define required field names as class constant
    _REQUIRED_FIELDS = {"agent_step", "epoch", "generation", "train_time"}

    # Type hints for IDE support
    agent_step: int
    epoch: int
    generation: int
    train_time: float

    def __init__(self, agent_step=0, epoch=0, generation=0, train_time=0.0, **kwargs: Any):
        """Initialize with required fields and optional additional fields."""

        # Build the data dict, letting kwargs override defaults
        data = {
            "agent_step": agent_step,
            "epoch": epoch,
            "generation": generation,
            "train_time": train_time,
        }

        overrides = set(data.keys()) & set(kwargs.keys())
        if overrides:
            logger.warning(f"kwargs override positional arguments: {overrides}")

        data.update(kwargs)
        super().__init__(data)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access for any field in the dict."""
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-style setting for any field."""
        if name.startswith("_"):
            # Allow setting private attributes normally
            super().__setattr__(name, value)
        else:
            # Set any field in the dict
            self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete fields via attribute access, but prevent deletion of required fields."""
        if name in self._REQUIRED_FIELDS:
            raise AttributeError(f"Cannot delete required field: {name}")
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

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
