from pathlib import Path
from typing import Any, ClassVar, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="ConfigWithBuilder")


class TypeSafeBuilder:
    """A type-safe builder for Pydantic models."""

    def __init__(self, model_class: Type[T]):
        self._model_class = model_class
        self._data: dict[str, Any] = {}

        # Dynamically add setter methods for each field
        for field_name in model_class.model_fields:
            setattr(self, field_name, self._create_setter(field_name))

    def _create_setter(self, field_name: str):
        def setter(value: Any) -> "TypeSafeBuilder":
            self._data[field_name] = value
            return self

        return setter

    def build(self) -> T:
        """Build and return the configured model instance."""
        return self._model_class(**self._data)


class ConfigWithBuilder(BaseModel):
    """Base class for configs that includes a type-safe builder pattern."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @classmethod
    def builder(cls: Type[T]) -> TypeSafeBuilder:
        """Return a type-safe builder for this config."""
        return TypeSafeBuilder(cls)

    @classmethod
    def from_file(cls: Type[T], config_path: str | Path) -> T:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls.model_validate(config_data)
