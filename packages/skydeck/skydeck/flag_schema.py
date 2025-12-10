"""Flag schema system for dynamic flag inference and typeahead."""

import json
from enum import Enum
from typing import Any, Optional, get_args, get_origin

from pydantic import BaseModel


class FlagType(str, Enum):
    """Type of a flag for UI rendering."""

    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    CHOICE = "CHOICE"  # Enum or literal


class FlagSchema(BaseModel):
    """Schema information for a single flag."""

    flag_name: str
    flag_type: FlagType
    description: Optional[str] = None
    default_value: Optional[str] = None
    choices: Optional[list[str]] = None  # For CHOICE type
    category: Optional[str] = None  # e.g., "trainer", "model"


def infer_flag_type(annotation: Any) -> tuple[FlagType, Optional[list[str]]]:
    """Infer flag type from Python type annotation.

    Args:
        annotation: Python type annotation

    Returns:
        Tuple of (FlagType, choices_list or None)
    """
    # Handle Optional types
    origin = get_origin(annotation)
    if origin is type(None) or str(origin) == "typing.Union":
        args = get_args(annotation)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            annotation = non_none_args[0]

    # Check if it's an Enum
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        choices = [item.value for item in annotation]
        return FlagType.CHOICE, choices

    # Check basic types
    if annotation is bool:
        return FlagType.BOOLEAN, None
    elif annotation is int:
        return FlagType.INTEGER, None
    elif annotation is float:
        return FlagType.FLOAT, None
    elif annotation is str:
        return FlagType.STRING, None

    # Default to string for unknown types
    return FlagType.STRING, None


def extract_flag_schemas_from_model(
    model: type[BaseModel], prefix: str = "", category: Optional[str] = None
) -> list[FlagSchema]:
    """Extract flag schemas from a Pydantic model recursively.

    Args:
        model: Pydantic model class to extract from
        prefix: Prefix for nested fields (e.g., "trainer.")
        category: Category name for grouping

    Returns:
        List of FlagSchema objects
    """
    schemas = []

    for field_name, field_info in model.model_fields.items():
        full_name = f"{prefix}{field_name}" if prefix else field_name
        annotation = field_info.annotation

        # Get description from field info
        description = field_info.description

        # Get default value
        default_value = None
        if field_info.default is not None and field_info.default is not ...:
            default_value = str(field_info.default)
        elif hasattr(field_info, "default_factory") and field_info.default_factory:
            # Can't easily get default from factory, skip
            pass

        # Infer type
        flag_type, choices = infer_flag_type(annotation)

        # Check if this is a nested model
        origin = get_origin(annotation)
        if origin is type(None) or str(origin) == "typing.Union":
            args = get_args(annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                annotation = non_none_args[0]

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # Recursively extract from nested model
            nested_schemas = extract_flag_schemas_from_model(
                annotation, prefix=f"{full_name}.", category=category or field_name
            )
            schemas.extend(nested_schemas)
        else:
            # Add schema for this field
            schema = FlagSchema(
                flag_name=full_name,
                flag_type=flag_type,
                description=description,
                default_value=default_value,
                choices=choices,
                category=category or prefix.rstrip(".") if prefix else None,
            )
            schemas.append(schema)

    return schemas


async def store_flag_schemas(db, schemas: list[FlagSchema]):
    """Store flag schemas in database.

    Args:
        db: Database instance
        schemas: List of FlagSchema objects to store
    """
    for schema in schemas:
        await db._conn.execute(
            """
            INSERT OR REPLACE INTO flag_schemas
            (flag_name, flag_type, description, default_value, choices, category)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                schema.flag_name,
                schema.flag_type.value,
                schema.description,
                schema.default_value,
                json.dumps(schema.choices) if schema.choices else None,
                schema.category,
            ),
        )
    await db._conn.commit()


async def get_flag_schemas(db) -> list[FlagSchema]:
    """Get all flag schemas from database.

    Args:
        db: Database instance

    Returns:
        List of FlagSchema objects
    """
    cursor = await db._conn.execute("SELECT * FROM flag_schemas ORDER BY flag_name")
    rows = await cursor.fetchall()

    schemas = []
    for row in rows:
        schema = FlagSchema(
            flag_name=row["flag_name"],
            flag_type=FlagType(row["flag_type"]),
            description=row["description"],
            default_value=row["default_value"],
            choices=json.loads(row["choices"]) if row["choices"] else None,
            category=row["category"],
        )
        schemas.append(schema)

    return schemas


# Example usage for populating schemas from your config models
# This would typically be run at startup or when config changes


def populate_schemas_from_config_models():
    """Example function showing how to populate schemas.

    You would import your actual config Pydantic models and call
    extract_flag_schemas_from_model on them.
    """
    # Example:
    # from your_package.config import TrainerConfig, ModelConfig
    # schemas = []
    # schemas.extend(extract_flag_schemas_from_model(TrainerConfig, prefix="trainer."))
    # schemas.extend(extract_flag_schemas_from_model(ModelConfig, prefix="model."))
    # return schemas
    pass
