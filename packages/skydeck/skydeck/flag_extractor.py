"""Extract flag definitions from Tool classes using pydantic introspection."""

import inspect
import logging
from typing import Any

from pydantic import BaseModel

from metta.common.tool import Tool
from metta.common.tool.tool_path import resolve_and_load_tool_maker

logger = logging.getLogger(__name__)


def get_pydantic_field_info(
    model_class: type[BaseModel], prefix: str = ""
) -> list[tuple[str, str, Any, bool]]:
    """Recursively get field information from a Pydantic model.

    Returns list of (path, type_str, default, required) tuples.
    This is copied from run_tool.py to keep skydeck self-contained.
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
                actual_type = (
                    non_none_types[0] if len(non_none_types) == 1 else annotation
                )
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
                    default_val = (
                        field.default_factory if field.default_factory else None
                    )
                is_required = (
                    field.is_required()
                    if hasattr(field, "is_required")
                    else (default_val is None)
                )
                fields_info.append((field_path, type_name, default_val, is_required))

                # Then recursively get nested fields
                nested_fields = get_pydantic_field_info(actual_type, field_path)
                fields_info.extend(nested_fields)
            else:
                # Regular field
                type_name = getattr(actual_type, "__name__", str(actual_type))
                default_val = (
                    field.default if field.default is not None else field.default_factory
                )
                is_required = (
                    field.is_required()
                    if hasattr(field, "is_required")
                    else (default_val is None)
                )
                fields_info.append((field_path, type_name, default_val, is_required))
        except (TypeError, AttributeError):
            # For complex types that can't be inspected
            type_name = str(annotation).replace("typing.", "")
            default_val = (
                field.default if field.default is not None else field.default_factory
            )
            is_required = (
                field.is_required()
                if hasattr(field, "is_required")
                else (default_val is None)
            )
            fields_info.append((field_path, type_name, default_val, is_required))

    return fields_info


def extract_flags_from_tool_path(tool_path: str) -> list[dict[str, Any]]:
    """Extract all flag definitions from a tool class.

    Args:
        tool_path: Full or short tool path (e.g., "arena.train" or "recipes.experiment.arena.train")

    Returns:
        List of flag definitions with keys: flag, type, default, required

    Example:
        >>> extract_flags_from_tool_path("arena.train")
        [
            {"flag": "trainer.batch_size", "type": "int", "default": 1024, "required": False},
            {"flag": "trainer.learning_rate", "type": "float", "default": 0.0003, "required": False},
            ...
        ]
    """
    try:
        # Resolve and load the tool maker
        tool_maker = resolve_and_load_tool_maker(tool_path)

        if tool_maker is None:
            logger.warning(f"Could not resolve tool path: {tool_path}")
            return []

        # Only extract from Tool classes, not functions
        if not (inspect.isclass(tool_maker) and issubclass(tool_maker, Tool)):
            logger.warning(f"Tool path {tool_path} is not a Tool class: {tool_maker}")
            return []

        # Extract field information
        fields_info = get_pydantic_field_info(tool_maker)

        # Convert to flag definitions
        flags = []
        for path, type_str, default_val, required in fields_info:
            # Convert default value to a serializable format
            if callable(default_val):
                default_serializable = None  # Factory functions can't be serialized
            elif isinstance(default_val, BaseModel):
                default_serializable = None  # Complex objects shouldn't be serialized as defaults
            elif isinstance(default_val, str) and default_val.startswith("<"):
                default_serializable = None  # Skip placeholder values
            elif isinstance(default_val, (property, type)):
                default_serializable = None  # Skip property objects and type objects
            elif isinstance(default_val, (list, dict, set, tuple)):
                # Skip complex containers for now
                default_serializable = None
            elif isinstance(default_val, (str, int, float, bool, type(None))):
                default_serializable = default_val
            else:
                # For any other complex type, skip it
                default_serializable = None

            flags.append({
                "flag": path,
                "type": type_str,
                "default": default_serializable,
                "required": required,
            })

        return flags

    except Exception as e:
        logger.error(f"Failed to extract flags from {tool_path}: {e}", exc_info=True)
        return []


def get_default_flags() -> list[dict[str, Any]]:
    """Get flags from the default TrainTool.

    This is used as a fallback when tool_path is not specified for an experiment.
    """
    return extract_flags_from_tool_path("metta.tools.train.TrainTool")
