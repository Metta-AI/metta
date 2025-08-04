"""Type utilities for tAXIOM pipelines - MVP version."""

from __future__ import annotations

from typing import Any


def infer_type(func: Any) -> tuple[type | None, type | None]:
    """Attempt to infer input and output types from function annotations.
    
    Used for documentation purposes only - no runtime validation.

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
        # Skip 'self' parameter for methods
        if params[0].name != 'self':
            input_type = params[0].annotation
        elif len(params) > 1 and params[1].annotation != inspect.Parameter.empty:
            input_type = params[1].annotation

    # Infer output type from return annotation
    output_type = None
    if sig.return_annotation != inspect.Signature.empty:
        output_type = sig.return_annotation

    return input_type, output_type