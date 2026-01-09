"""Pydantic schema extraction utilities.

Used by run_tool.py for --help output and pydantic_config_schema.py for JSON export.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def get_type_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is None or annotation is type(None):
        return "null"

    origin = get_origin(annotation)

    # Handle Optional[X] / Union[X, None]
    if origin is Union:
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return get_type_str(non_none_args[0])
        return " | ".join(get_type_str(arg) for arg in non_none_args)

    # Handle list, dict, etc.
    if origin is list:
        args = get_args(annotation)
        if args:
            return f"list[{get_type_str(args[0])}]"
        return "list"

    if origin is dict:
        args = get_args(annotation)
        if args and len(args) == 2:
            return f"dict[{get_type_str(args[0])}, {get_type_str(args[1])}]"
        return "dict"

    # Handle Literal
    if str(origin) == "typing.Literal":
        args = get_args(annotation)
        return f"Literal{list(args)}"

    # Handle Enum
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        choices = [item.value for item in annotation]
        return f"Enum{choices}"

    # Handle basic types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Fallback
    return str(annotation).replace("typing.", "")


def serialize_default(value: Any) -> Any:
    """Serialize a default value to JSON-compatible format."""
    if value is None:
        return None
    # Handle PydanticUndefined
    if str(type(value).__name__) == "PydanticUndefinedType":
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [serialize_default(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_default(v) for k, v in value.items()}
    if isinstance(value, BaseModel):
        return f"<{value.__class__.__name__}>"
    if callable(value):
        return "<factory>"
    # Handle timedelta, Path, etc.
    return str(value)


def get_pydantic_field_info(model_class: type[BaseModel], prefix: str = "") -> list[tuple[str, str, Any, bool]]:
    """Recursively get field information from a Pydantic model.

    Args:
        model_class: Pydantic model class to extract from
        prefix: Prefix for nested field paths (e.g., "trainer.")

    Returns:
        List of (path, type_str, default, required) tuples where:
        - path: Dotted path like "trainer.batch_size"
        - type_str: Human-readable type string
        - default: Default value (may be callable for factories)
        - required: Whether the field is required
    """
    fields_info = []

    for field_name, field in model_class.model_fields.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        annotation = field.annotation

        # Get the origin type if it's a generic
        origin = getattr(annotation, "__origin__", None)

        # Handle Optional types
        if origin is type(None):
            actual_type = annotation
        elif hasattr(annotation, "__args__"):
            # For Optional[X], Union[X, None], etc.
            args = getattr(annotation, "__args__", ())
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                actual_type = non_none_types[0] if len(non_none_types) == 1 else annotation
            else:
                actual_type = annotation
        else:
            actual_type = annotation

        # Check if it's a nested Pydantic model
        try:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
                # Add the parent field first (before nested fields for better ordering)
                type_name = actual_type.__name__
                # Don't show the full default object representation for complex models
                if field.default is not None and not callable(field.default):
                    default_val = f"<{type_name} instance>"
                else:
                    default_val = field.default_factory if field.default_factory else None
                is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
                fields_info.append((field_path, type_name, default_val, is_required))

                # Then recursively get nested fields
                nested_fields = get_pydantic_field_info(actual_type, field_path)
                fields_info.extend(nested_fields)
            else:
                # Regular field
                type_name = getattr(actual_type, "__name__", str(actual_type))
                default_val = field.default if field.default is not None else field.default_factory
                is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
                fields_info.append((field_path, type_name, default_val, is_required))
        except (TypeError, AttributeError):
            # For complex types that can't be inspected
            type_name = str(annotation).replace("typing.", "")
            default_val = field.default if field.default is not None else field.default_factory
            is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
            fields_info.append((field_path, type_name, default_val, is_required))

    return fields_info


def extract_schema(
    model_class: type[BaseModel],
    prefix: str = "",
    include_nested_models: bool = True,
) -> dict[str, dict[str, Any]]:
    """Recursively extract schema from a Pydantic model as a dictionary.

    This is a richer version of get_pydantic_field_info that returns a dict
    suitable for JSON serialization.

    Args:
        model_class: Pydantic model class to extract from
        prefix: Prefix for nested field paths
        include_nested_models: Whether to recurse into nested models

    Returns:
        Dict mapping field paths to {type, default, required, description}
    """
    schema: dict[str, dict[str, Any]] = {}

    for field_name, field in model_class.model_fields.items():
        field_path = f"{prefix}{field_name}" if prefix else field_name
        annotation = field.annotation

        # Unwrap Optional
        origin = get_origin(annotation)
        actual_type = annotation
        if origin is Union:
            args = get_args(annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                actual_type = non_none_args[0]

        # Get default value
        default_val = None
        has_default = False

        # Check for PydanticUndefined
        if field.default is not None and field.default is not ...:
            if str(type(field.default).__name__) != "PydanticUndefinedType":
                default_val = field.default
                has_default = True

        if not has_default and field.default_factory is not None:
            try:
                default_val = field.default_factory()
                has_default = True
            except Exception:
                default_val = "<factory>"
                has_default = True

        # Check if required
        is_required = field.is_required() if hasattr(field, "is_required") else (not has_default)

        # Check if nested Pydantic model
        is_nested_model = False
        try:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
                is_nested_model = True
        except TypeError:
            pass

        if is_nested_model and include_nested_models:
            # Add entry for the nested model itself
            schema[field_path] = {
                "type": actual_type.__name__,
                "default": serialize_default(default_val) if default_val else None,
                "required": is_required,
            }
            if field.description:
                schema[field_path]["description"] = field.description

            # Recurse into nested model
            nested_schema = extract_schema(actual_type, prefix=f"{field_path}.")
            schema.update(nested_schema)
        else:
            # Regular field
            entry: dict[str, Any] = {
                "type": get_type_str(annotation),
                "default": serialize_default(default_val),
                "required": is_required,
            }
            if field.description:
                entry["description"] = field.description

            schema[field_path] = entry

    return schema
