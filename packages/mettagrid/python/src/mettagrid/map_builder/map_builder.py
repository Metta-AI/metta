from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, Type, TypeVar

import yaml
from pydantic import (
    ModelWrapValidatorHandler,
    SerializeAsAny,
    model_serializer,
    model_validator,
)

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

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    @classmethod
    def builder_cls(cls) -> Type[TBuilder]:
        if cls._builder_cls is None:
            raise TypeError(f"{cls.__class__.__name__} is not bound to a MapBuilder")
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
        builder_config = cls.model_validate(parsed)

        return builder_config

    # Ensure YAML/JSON dumps always include a 'type' with a nice FQCN
    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)  # dict of the model's fields

        type_cls = self.builder_cls()
        type_str = f"{type_cls.__module__}.{type_cls.__name__}"

        return {"type": type_str, **data}

    @model_validator(mode="wrap")
    @classmethod
    def _validate_with_type(
        cls, v: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """
        Accepts any of:
        - a MapBuilderConfig instance (already specific)
        - a dict with {"type": "<FQCN-of-Builder-or-Config>", ...params...}
        """
        if isinstance(v, MapBuilderConfig):
            if not isinstance(v, cls):
                raise TypeError(
                    f"Expected {cls.__qualname__}, got {type(v).__qualname__}"
                )
            return v

        if not isinstance(v, dict):
            raise ValueError("MapBuilder config must be a dict")

        t = v.get("type")
        if t is None:
            # Valid when instantiated from Python, e.g. `AsciiMapBuilder.Config(...)` won't include `type`.
            return handler(v)

        # Import the symbol named in 'type'
        type_cls = load_symbol(t) if isinstance(t, str) else t

        if not inspect.isclass(type_cls):
            raise TypeError("'type' must point to a class")

        # If it's a MapBuilder, use its nested Config
        if not issubclass(type_cls, MapBuilder):
            raise TypeError(
                f"'type' must point to a MapBuilder subclass; got {type_cls.__qualname__}"
            )

        cfg_cls = getattr(type_cls, "Config", None)
        if not (isinstance(cfg_cls, type) and issubclass(cfg_cls, MapBuilderConfig)):
            raise TypeError(
                f"{type_cls.__qualname__} must define a nested class Config(MapBuilderConfig)."
            )

        # `cfg_cls` can be more specific than `cls`.
        # This might matter when we load the config from YAML through a specific MapBuilderConfig subclass.
        # For example, `AsciiMapBuilder.Config.from_uri()` will return an instance of `AsciiMapBuilder.Config`.
        if not issubclass(cfg_cls, cls):
            raise TypeError(
                f"'type' {cfg_cls.__qualname__} is not a subclass of {cls.__qualname__}"
            )

        data = {k: v for k, v in v.items() if k != "type"}
        result = cfg_cls.model_validate(data)

        assert isinstance(
            result, cls
        )  # should always be true because we checked the subclass relationship above

        return result


AnyMapBuilderConfig = SerializeAsAny[MapBuilderConfig]


class MapBuilder(ABC):
    """
    A base class for building MettaGridEnv game maps.

    If a subclass declares a nested class `Config` that inherits from MapBuilderConfig, it will be *automatically
    bound*.
    """

    Config: ClassVar[type[MapBuilderConfig]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.Config._builder_cls = cls  # type: ignore[assignment]

    @abstractmethod
    def build(self) -> GameMap: ...
