from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Generic, List, Optional, Set, Tuple, Type, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import SerializeAsAny, WrapValidator, model_serializer, model_validator

from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol

# We store maps as 2D arrays of object names.
# "empty" means an empty cell; "wall" means a wall, etc. See `metta.mettagrid.char_encoder` for the full list.
#
# Properly shaped version, `np.ndarray[tuple[int, int], np.dtype[np.str_]]`,
# would be better, but slices from numpy arrays are not typed properly, which makes it too annoying to use.


MapGrid: TypeAlias = npt.NDArray[np.str_]

map_grid_dtype = np.dtype("<U20")


class GameMap:
    """
    Represents a game map in the MettaGrid game.
    """

    # Two-dimensional grid of strings.
    # Possible values: "wall", "empty", "agent", etc.
    # For the full list, see `mettagrid_c.cpp`.
    grid: MapGrid

    # Optional compressed representation
    byte_grid: Optional[np.ndarray] = None
    object_key: Optional[List[str]] = None

    def __init__(self, grid: MapGrid, byte_grid=None, object_key=None):
        self.grid = grid
        self.byte_grid = byte_grid
        self.object_key = object_key

    def compress(self, valid_objects: Optional[Set[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """Compress the string grid to byte grid + key."""
        if self.byte_grid is not None and self.object_key is not None:
            return self.byte_grid, self.object_key

        from metta.mettagrid.mapgen.utils.map_compression import MapCompressor

        compressor = MapCompressor(valid_objects)
        self.byte_grid, self.object_key = compressor.compress(self.grid)
        return self.byte_grid, self.object_key


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

    # Optional game config for validation
    game_config: Optional[Any] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.Config._builder_cls = cls  # type: ignore[assignment]

    def set_game_config(self, game_config) -> None:
        """Set game config for validation (optional)."""
        self.game_config = game_config

    def get_valid_objects(self) -> Optional[Set[str]]:
        """Get set of valid object names from game config."""
        if self.game_config and hasattr(self.game_config, "objects"):
            return set(self.game_config.objects.keys())
        return None

    @abstractmethod
    def build(self) -> GameMap: ...

    def build_validated(self) -> GameMap:
        """Build with validation against game config."""
        game_map = self.build()

        # Validate if game config is available
        valid_objects = self.get_valid_objects()
        if valid_objects:
            try:
                game_map.compress(valid_objects)
            except ValueError as e:
                # Log warning but don't fail (backward compatibility)
                import logging

                logging.warning(f"Map validation warning: {e}")

        return game_map


def _validate_open_map_builder(v: Any, handler):
    """
    Accepts any of:
      - a MapBuilderConfig instance (already specific)
      - a dict with {"type": "<FQCN-of-Builder-or-Config>", ...params...}
      - anything else -> let the default handler try (will error if invalid)
    """
    if isinstance(v, MapBuilderConfig):
        return v

    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            # try default handler first (e.g., if the default type is already implied)
            return handler(v)

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

    # Fallback to the normal validator (will raise a decent error)
    return handler(v)


AnyMapBuilderConfig = SerializeAsAny[Annotated[MapBuilderConfig[Any], WrapValidator(_validate_open_map_builder)]]

C = TypeVar("C")
