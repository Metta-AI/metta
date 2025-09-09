from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    List,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from pydantic import SerializeAsAny, WrapValidator, model_serializer, model_validator

from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_config import GameConfig

# Maps can be stored as either:
# 1. Legacy format: 2D arrays of object names (strings)
# 2. New format: 2D arrays of type IDs (uint8) with decoder key
#
# During migration, we support both formats. New format provides:
# - ~95% memory savings (1 byte vs 20 bytes per cell)
# - Better performance (int comparisons vs string comparisons)
# - Type safety through constants

# Legacy string-based format
MapGridLegacy: TypeAlias = npt.NDArray[np.str_]
map_grid_legacy_dtype = np.dtype("<U20")

# New int-based format
MapGridInt: TypeAlias = npt.NDArray[np.uint8]
map_grid_int_dtype = np.dtype(np.uint8)

# Union type for backward compatibility during migration
MapGrid: TypeAlias = Union[MapGridLegacy, MapGridInt]


class GameMap:
    """
    Represents a game map in the MettaGrid game.

    Supports both legacy string-based format and new int-based format
    during migration period. New format includes decoder key for
    converting type IDs back to human-readable names.
    """

    def __init__(self, grid: MapGrid, decoder_key: Optional[List[str]] = None):
        """
        Initialize GameMap with grid data.

        Args:
            grid: Either legacy string grid or new int-based grid
            decoder_key: Required for int-based grids. Maps type_id -> object_name
        """
        self.grid = grid
        self.decoder_key = decoder_key

        # Validate format consistency
        if self.is_int_based():
            if decoder_key is None:
                raise ValueError("decoder_key required for int-based grids")
            self._validate_int_grid()
        elif decoder_key is not None:
            # String grid with decoder key - convert to int format
            raise ValueError("decoder_key should not be provided for string-based grids")

    def is_int_based(self) -> bool:
        """Check if this map uses the new int-based format."""
        return self.grid.dtype == np.uint8

    def is_legacy(self) -> bool:
        """Check if this map uses the legacy string-based format."""
        return self.grid.dtype.kind in ("U", "S")  # Unicode or byte strings

    def _validate_int_grid(self):
        """Validate int-based grid has valid type IDs."""
        if self.decoder_key is None:
            return

        max_type_id = len(self.decoder_key) - 1
        grid_max = np.max(self.grid)

        if int(grid_max) > max_type_id:
            raise ValueError(
                f"Grid contains type_id {grid_max} but decoder_key only has {len(self.decoder_key)} entries"
            )

    def get_object_name(self, row: int, col: int) -> str:
        """Get object name at grid position, handling both formats.

        Args:
            row: Grid row index
            col: Grid column index

        Returns:
            Object name at the position
        """
        if self.is_legacy():
            return str(self.grid[row, col])
        else:
            type_id = int(self.grid[row, col])
            if self.decoder_key and 0 <= type_id < len(self.decoder_key):
                return self.decoder_key[type_id]
            else:
                return f"unknown_type_{type_id}"

    def get_type_id(self, row: int, col: int) -> int:
        """Get type ID at grid position, handling both formats.

        Args:
            row: Grid row index
            col: Grid column index

        Returns:
            Type ID at the position (0 for unknown objects in legacy format)
        """
        if self.is_int_based():
            return int(self.grid[row, col])
        else:
            # Legacy format - would need type mapping to convert
            # For now, return 0 for empty, 1 for everything else
            obj_name = str(self.grid[row, col])
            return 0 if obj_name == "empty" else 1

    @property
    def shape(self):
        """Get shape of the grid."""
        return self.grid.shape

    def to_legacy_format(self) -> "GameMap":
        """Convert to legacy string format (for backward compatibility)."""
        if self.is_legacy():
            return GameMap(self.grid.copy())

        # Convert int grid to string grid using decoder
        if self.decoder_key is None:
            raise ValueError("Cannot convert to legacy format without decoder_key")

        string_grid = np.full(self.grid.shape, "unknown", dtype=map_grid_legacy_dtype)
        for type_id, obj_name in enumerate(self.decoder_key):
            if obj_name:  # Skip empty entries
                mask = self.grid == type_id
                string_grid[mask] = obj_name

        return GameMap(string_grid)

    def get_legacy_grid(self) -> MapGridLegacy:
        """Get grid in legacy string format (for C++ interface compatibility)."""
        if self.is_legacy():
            # Type guard: we know this is MapGridLegacy based on is_legacy check
            return self.grid  # type: ignore[return-value]
        else:
            return self.to_legacy_format().grid  # type: ignore[return-value]


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


def create_int_grid(height: int, width: int, fill_type_id: int = 0) -> MapGridInt:
    """Create a new int-based grid filled with the specified type ID.

    Args:
        height: Grid height
        width: Grid width
        fill_type_id: Type ID to fill the grid with (default 0 for empty)

    Returns:
        New int-based grid
    """
    return np.full((height, width), fill_type_id, dtype=map_grid_int_dtype)


def create_legacy_grid(height: int, width: int, fill_value: str = "empty") -> MapGridLegacy:
    """Create a new legacy string-based grid filled with the specified value.

    Args:
        height: Grid height
        width: Grid width
        fill_value: String value to fill the grid with

    Returns:
        New legacy string-based grid
    """
    return np.full((height, width), fill_value, dtype=map_grid_legacy_dtype)


class MapBuilder(ABC):
    """
    A base class for building MettaGridEnv game maps.

    If a subclass declares a nested class `Config` that inherits from MapBuilderConfig, it will be *automatically
    bound*.

    During migration period, supports both legacy (string-based) and new (int-based) map generation.
    New builders should use GameConfig parameterization for type validation and int-based output.
    """

    Config: ClassVar[type[MapBuilderConfig[Any]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.Config._builder_cls = cls  # type: ignore[assignment]

    @abstractmethod
    def build(self) -> GameMap: ...

    def supports_int_format(self) -> bool:
        """
        Override in subclasses to indicate support for int-based format.

        Returns:
            True if the builder supports generating int-based maps
        """
        return False

    def supports_game_config_param(self) -> bool:
        """
        Override in subclasses to indicate support for GameConfig parameterization.

        Returns:
            True if the builder accepts GameConfig for validation and type mapping
        """
        return False


class EnhancedMapBuilder(MapBuilder):
    """
    Enhanced MapBuilder with GameConfig parameterization and int-based map support.

    This is the new base class that builders should inherit from to support:
    - GameConfig parameterization for object validation
    - Int-based map generation for memory efficiency
    - Type mapping utilities

    During migration, existing builders can gradually move to this base class.
    """

    class Config(MapBuilderConfig["EnhancedMapBuilder"]):
        pass

    def __init__(self, config: MapBuilderConfig[Any], game_config: Optional["GameConfig"] = None):
        """
        Initialize enhanced map builder.

        Args:
            config: Map builder configuration
            game_config: Optional game configuration for type validation and mapping
        """
        from metta.mettagrid.type_mapping import TypeMapping

        self.config = config
        self.game_config = game_config

        # Set up type mapping based on GameConfig if provided
        if game_config:
            self.type_mapping = TypeMapping(game_config)
        else:
            self.type_mapping = TypeMapping()  # Use standard mappings

    def supports_int_format(self) -> bool:
        """Enhanced builders support int format by default."""
        return True

    def supports_game_config_param(self) -> bool:
        """Enhanced builders support GameConfig parameterization."""
        return True

    def get_type_id(self, obj_name: str) -> int:
        """
        Get type ID for object name using type mapping.

        Args:
            obj_name: Object name to look up

        Returns:
            Type ID for the object

        Raises:
            KeyError: If object name is not found
        """
        return self.type_mapping.get_type_id(obj_name)

    def validate_object_availability(self, obj_name: str) -> bool:
        """
        Validate that an object is available in the current GameConfig.

        Args:
            obj_name: Object name to validate

        Returns:
            True if object is available, False otherwise
        """
        if not self.game_config:
            # No GameConfig - allow standard objects
            return self.type_mapping.has_name(obj_name)

        # Check if object exists in GameConfig
        return obj_name in self.game_config.objects or obj_name == "empty"

    def create_int_map(self, height: int, width: int, fill_type_id: int = 0) -> GameMap:
        """
        Create a new int-based GameMap.

        Args:
            height: Map height
            width: Map width
            fill_type_id: Type ID to fill with (default 0 for empty)

        Returns:
            New GameMap with int-based grid and decoder key
        """
        grid = create_int_grid(height, width, fill_type_id)
        decoder_key = self.type_mapping.get_decoder_key()
        return GameMap(grid, decoder_key)

    def create_legacy_map(self, height: int, width: int, fill_value: str = "empty") -> GameMap:
        """
        Create a new legacy string-based GameMap.

        Args:
            height: Map height
            width: Map width
            fill_value: String value to fill with

        Returns:
            New GameMap with legacy string-based grid
        """
        grid = create_legacy_grid(height, width, fill_value)
        return GameMap(grid)


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
