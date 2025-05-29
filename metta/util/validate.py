from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Union, get_args, get_origin, get_type_hints


def _is_instance(value: Any, expected_type: Any) -> bool:
    origin = get_origin(expected_type)
    if origin is Union:
        return any(_is_instance(value, arg) for arg in get_args(expected_type))

    if origin is not None:
        expected_type = origin

    try:
        return isinstance(value, expected_type)
    except TypeError:
        return False


def validate_arg_types(func):
    """Assert argument types based on annotations."""

    hints = get_type_hints(func)
    sig = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            expected = hints.get(name)
            if expected is not None:
                assert _is_instance(value, expected), f"Argument '{name}' expected {expected} but got {type(value)}"
        return func(*args, **kwargs)

    return wrapper
