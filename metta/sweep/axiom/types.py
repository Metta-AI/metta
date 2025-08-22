"""Type validation and utilities for tAXIOM pipelines."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Union, get_args, get_origin, runtime_checkable

from pydantic import BaseModel

# Protocols for marking types as having specific capabilities


@runtime_checkable
class Loggable(Protocol):
    """Protocol for types that can be logged to external systems."""

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        ...


# Decorator for marking types as loggable


def loggable(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    custom_serializers: dict[type, Callable] | None = None,
):
    """Decorator to mark a class as loggable with fine-grained control.

    Args:
        include: List of field names to include (None means all)
        exclude: List of field names to exclude
        custom_serializers: Dict mapping types to serialization functions

    Usage:
        @loggable()
        class MyClass: ...

        @loggable(exclude=["sensitive_data"])
        class SecureClass: ...

        @loggable(custom_serializers={np.ndarray: lambda x: x.tolist()})
        class NumpyClass: ...
    """

    def decorator(cls):
        def to_log_dict(self) -> dict[str, Any]:
            # Start with base data
            if hasattr(self, "model_dump"):  # Pydantic model
                data = self.model_dump()
            elif hasattr(self, "__dict__"):  # Regular class
                data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            else:
                data = {}

            # Apply include filter
            if include is not None:
                data = {k: v for k, v in data.items() if k in include}

            # Apply exclude filter
            if exclude is not None:
                data = {k: v for k, v in data.items() if k not in exclude}

            # Apply custom serializers
            if custom_serializers:
                for key, value in data.items():
                    for type_key, serializer in custom_serializers.items():
                        if isinstance(value, type_key):
                            data[key] = serializer(value)
                            break

            return data

        # Add the to_log_dict method
        cls.to_log_dict = to_log_dict

        # Mark class as implementing Loggable protocol
        cls.__loggable__ = True

        return cls

    # Handle both @loggable and @loggable() syntax
    if callable(include):
        # Called as @loggable without parentheses
        cls = include
        include = None
        return decorator(cls)
    else:
        # Called as @loggable() with arguments
        return decorator


# Validation utilities


def validate_type(data: Any, expected_type: type) -> None:
    """Validate that data matches expected type.

    Handles:
    - Pydantic models (via model_validate)
    - Runtime protocols (via isinstance)
    - Standard types (via isinstance)
    - Generic types (list[T], dict[K, V], etc.)
    - Union types (Type1 | Type2)
    """
    if expected_type is None:
        return

    # Handle Pydantic models
    if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
        try:
            if isinstance(data, expected_type):
                # Already the right type, just validate
                data.model_validate(data.model_dump())
            else:
                # Try to construct and validate
                expected_type.model_validate(data)
        except Exception as e:
            raise TypeError(f"Validation failed for {expected_type.__name__}: {e}") from e
        return

    # Handle Union types (Type1 | Type2 or Union[Type1, Type2])
    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        for arg in args:
            try:
                validate_type(data, arg)
                return  # Success with this type
            except TypeError:
                continue
        # None of the union types matched
        raise TypeError(f"Expected one of {args}, got {type(data)}")

    # Handle Generic types
    if origin is not None:
        # Check the container type first
        if not isinstance(data, origin):
            raise TypeError(f"Expected {expected_type}, got {type(data)}")

        # For common generics, validate contents
        args = get_args(expected_type)
        if origin is list and args:
            for item in data:
                validate_type(item, args[0])
        elif origin is dict and len(args) == 2:
            for key, value in data.items():
                validate_type(key, args[0])
                validate_type(value, args[1])
        elif origin is tuple and args:
            if len(args) == 2 and args[1] is ...:  # tuple[T, ...]
                for item in data:
                    validate_type(item, args[0])
            else:  # tuple[T1, T2, T3]
                if len(data) != len(args):
                    raise TypeError(f"Expected tuple of length {len(args)}, got {len(data)}")
                for item, expected in zip(data, args, strict=False):
                    validate_type(item, expected)
        return

    # Handle Protocol types
    if hasattr(expected_type, "__protocol__"):
        if not isinstance(data, expected_type):
            raise TypeError(f"Object does not implement protocol {expected_type.__name__}")
        return

    # Standard type checking
    if not isinstance(data, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(data).__name__}")


def infer_type(func: Any) -> tuple[type | None, type | None]:
    """Attempt to infer input and output types from function annotations.

    Returns:
        (input_type, output_type) tuple, with None for any that can't be inferred
    """
    if not callable(func):
        return None, None

    import inspect

    sig = inspect.signature(func)

    # Infer input type from first parameter
    input_type = None
    params = list(sig.parameters.values())
    if params and params[0].annotation != inspect.Parameter.empty:
        input_type = params[0].annotation

    # Infer output type from return annotation
    output_type = None
    if sig.return_annotation != inspect.Signature.empty:
        output_type = sig.return_annotation

    return input_type, output_type
