"""Configuration protocols for environment-agnostic curriculum system."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = ["TaskConfig", "TConfig"]


@runtime_checkable
class TaskConfig(Protocol):
    """
    Protocol defining the interface for task configurations.

    Any configuration class used with agora must implement these methods.
    Pydantic BaseModel classes automatically satisfy this protocol.

    Example:
        >>> from pydantic import BaseModel
        >>> class MyConfig(BaseModel):
        ...     difficulty: int = 1
        >>> # MyConfig automatically satisfies TaskConfig protocol
    """

    def model_copy(self, *, deep: bool = False) -> TaskConfig:
        """
        Create a copy of this configuration.

        Args:
            deep: If True, create a deep copy with all nested objects duplicated

        Returns:
            A copy of the configuration
        """
        ...

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
    ) -> dict[str, Any]:
        """
        Serialize configuration to dictionary.

        Args:
            mode: Serialization mode ('python' or 'json')
            include: Fields to include
            exclude: Fields to exclude
            by_alias: Use field aliases
            exclude_unset: Exclude unset fields
            exclude_defaults: Exclude fields with default values
            exclude_none: Exclude None values
            round_trip: Enable round-trip serialization
            warnings: Enable warnings

        Returns:
            Dictionary representation of configuration
        """
        ...

    @classmethod
    def model_validate(cls, obj: Any) -> TaskConfig:
        """
        Validate and create configuration from dictionary or object.

        Args:
            obj: Dictionary or object to validate

        Returns:
            Validated configuration instance

        Raises:
            ValidationError: If validation fails
        """
        ...


# Generic type variable bound to TaskConfig protocol
# Used throughout agora to provide type-safe generic curriculum components
TConfig = TypeVar("TConfig", bound=TaskConfig)
