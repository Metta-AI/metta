from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, Self, Type, TypeVar

import yaml
from pydantic import SerializeAsAny, WrapValidator, model_serializer, model_validator

from mettagrid.base_config import Config
from mettagrid.mapgen.types import MapGrid
from mettagrid.util.module import load_symbol


class GameMap:
    """
    Represents a game map in the MettaGrid game.
    """

    # Two-dimensional grid of strings.
    # Possible values: "wall", "empty", "agent", etc.
    # For the full list, see `mettagrid_c.cpp`.
    grid: MapGrid

    def __init__(self, grid: MapGrid):
        self.grid = grid


TBuilder = TypeVar("TBuilder", bound="MapBuilder")


class MapBuilderConfig(Config, Generic[TBuilder]):
    """
    Base class for all map builder configs. Subclasses *optionally* know
    which MapBuilder they build via `_builder_cls` (auto-filled when nested
    inside a MapBuilder subclass; see MapBuilder.__init_subclass__ below).
    """

    _builder_cls: ClassVar = None

    def create(self) -> TBuilder:
        """
        Instantiate the bound MapBuilder.
        Subclasses nested under a MapBuilder automatically bind `_builder_cls`.
        If you define a standalone Config subclass, either set `_builder_cls`
        on the class or override `create()`.
        """
        return self.builder_cls()(self)  # type: ignore[call-arg]

    @classmethod
    def builder_cls(cls) -> Type[TBuilder]:
        if cls._builder_cls is None:
            raise TypeError(
                f"{cls.__class__.__name__} is not bound to a MapBuilder; "
                f"either define it nested under the builder or set `_builder_cls`."
            )
        return cls._builder_cls

    @classmethod
    def from_uri(cls, uri: str | Path) -> Self:
        """Load a builder config from a YAML or JSON file."""

        path = Path(uri)
        with path.open("r", encoding="utf-8") as f:
            raw = f.read()

        return cls.from_str(raw)

    @classmethod
    def from_str(cls, data: str | bytes) -> Self:
        """Load a builder config from a serialized string or mapping."""

        parsed = yaml.safe_load(data)
        builder_config = validate_any_map_builder(parsed)

        # Runtime type check to ensure we return the correct type
        if not isinstance(builder_config, cls):
            raise TypeError(f"Expected {cls.__name__} instance, got {type(builder_config).__name__}")

        return builder_config

    @model_validator(mode="before")
    @classmethod
    def _strip_type_field(cls, data: Any) -> Any:
        """Strip the 'type' field during validation if present."""
        if isinstance(data, dict) and "type" in data:
            builder_cls = cls.builder_cls()
            expected_type = f"{builder_cls.__module__}.{builder_cls.__name__}"
            if data["type"] != expected_type:
                raise ValueError(f"Invalid type: {data['type']}, expected {expected_type}")
            return {k: v for k, v in data.items() if k != "type"}
        return data

    # Ensure YAML/JSON dumps always include a 'type' with a nice FQCN
    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)  # dict of the model's fields
        typ_cls: Type[Any] = self._builder_cls or self.__class__
        # Prefer the *builder* class if known, fall back to the config class
        type_str = f"{typ_cls.__module__}.{typ_cls.__name__}"
        return {"type": type_str, **data}


class MapBuilder(ABC):
    """
    A base class for building MettaGridEnv game maps.

    If a subclass declares a nested class `Config` that inherits from MapBuilderConfig, it will be *automatically
    bound*.
    """

    Config: ClassVar[type[MapBuilderConfig[Any]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.Config._builder_cls = cls  # type: ignore[assignment]

    @abstractmethod
    def build(self) -> GameMap: ...


def validate_any_map_builder(v: Any):
    """
    Accepts any of:
      - a MapBuilderConfig instance (already specific)
      - a dict with {"type": "<FQCN-of-Builder-or-Config>", ...params...}
    """
    if isinstance(v, MapBuilderConfig):
        return v

    if not isinstance(v, dict):
        raise ValueError("MapBuilder config must be a dict")

    t = v.get("type")
    if t is None:
        raise ValueError("'type' is required")

    # Import the symbol named in 'type'
    target = load_symbol(t) if isinstance(t, str) else t

    # If it's a Builder, use its nested Config
    if isinstance(target, type) and issubclass(target, MapBuilder):
        cfg_model = getattr(target, "Config", None)
        if not (isinstance(cfg_model, type) and issubclass(cfg_model, MapBuilderConfig)):
            raise TypeError(f"{target.__name__} must define a nested class Config(MapBuilderConfig).")
        data = {k: v for k, v in v.items() if k != "type"}
        return cfg_model.model_validate(data)

    # If it's already a Config subclass, validate with it directly
    if isinstance(target, type) and issubclass(target, MapBuilderConfig):
        data = {k: v for k, v in v.items() if k != "type"}
        return target.model_validate(data)

    raise TypeError(f"'type' must point to a MapBuilder subclass or a MapBuilderConfig subclass; got {target!r}")


def _any_map_builder_wrap_validator(v: Any, handler):
    return validate_any_map_builder(v)


AnyMapBuilderConfig = SerializeAsAny[Annotated[MapBuilderConfig[Any], WrapValidator(_any_map_builder_wrap_validator)]]

C = TypeVar("C")
