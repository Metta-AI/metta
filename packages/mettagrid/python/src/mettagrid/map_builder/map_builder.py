from __future__ import annotations

import abc
import inspect
import logging
import pathlib
import typing

import numpy as np
import pydantic
import yaml

import mettagrid.base_config
import mettagrid.mapgen.types
import mettagrid.util.module

logger = logging.getLogger(__name__)


class GameMap:
    """
    Represents a game map in the MettaGrid game.
    """

    # Two-dimensional grid of strings.
    # Possible values: "wall", "empty", "agent", etc.
    # For the full list, see `mettagrid_c.cpp`.
    grid: mettagrid.mapgen.types.MapGrid

    def __init__(self, grid: mettagrid.mapgen.types.MapGrid):
        self.grid = grid


TBuilder = typing.TypeVar("TBuilder", bound="MapBuilder[typing.Any]")


class MapBuilderConfig(mettagrid.base_config.Config, typing.Generic[TBuilder]):
    """
    Base class for all map builder configs.
    """

    # Can't use correct generic parameter because ClassVar doesn't support generics.
    # We access this field through `builder_cls()` instead.
    _builder_cls: typing.ClassVar[type[MapBuilder] | None] = None

    @classmethod
    def builder_cls(cls) -> type[TBuilder]:
        if cls._builder_cls is None:
            raise TypeError(f"{cls.__qualname__} is not bound to a MapBuilder")
        return typing.cast(type[TBuilder], cls._builder_cls)

    def create(self) -> TBuilder:
        """
        Instantiate the bound MapBuilder.

        If your config class is generic over the builder class, this method will return the exact instance type.
        """
        return self.builder_cls()(self)  # type: ignore[call-arg]

    def model_dump(self, **kwargs) -> dict[str, typing.Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    @classmethod
    def from_uri(cls, uri: str | pathlib.Path) -> typing.Self:
        """Load a builder config from a YAML or JSON file."""

        path = pathlib.Path(uri)
        with path.open("r", encoding="utf-8") as f:
            raw = f.read()

        return cls.from_str(raw)

    @classmethod
    def from_str(cls, data: str | bytes) -> typing.Self:
        """Load a builder config from a serialized string or mapping."""

        parsed = yaml.safe_load(data)
        builder_config = cls.model_validate(parsed)

        return builder_config

    @classmethod
    def _type_str(cls) -> str:
        # Prefer builder_cls name (`RandomMapBuilder.Config`) over the original class name (`RandomMapBuilderConfig`).
        # This is important in case when the same config class is reused by multiple builders.
        # (See how `MapBuilder.__init_subclass__` clones the config class if it's already bound to another builder.)
        builder_cls = cls.builder_cls()
        return f"{builder_cls.__module__}.{builder_cls.__qualname__}.Config"

    # Ensure YAML/JSON dumps always include a 'type' with a nice FQCN
    @pydantic.model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)  # dict of the model's fields

        return {"type": self._type_str(), **data}

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _validate_with_type(
        cls, v: typing.Any, handler: pydantic.ModelWrapValidatorHandler[typing.Self]
    ) -> typing.Self:
        """
        Accepts any of:
        - a MapBuilderConfig instance (already specific)
        - a dict with {"type": "<FQCN-of-Config>", ...params...}
        """
        if isinstance(v, MapBuilderConfig):
            if not isinstance(v, cls):
                raise TypeError(f"Expected {cls.__qualname__} subclass, got {type(v).__qualname__}")
            return v

        if not isinstance(v, dict):
            raise ValueError("MapBuilderConfig params must be a dict")

        t = v.get("type")
        if t is None:
            # Valid when instantiated from Python, e.g. `AsciiMapBuilder.Config(...)` won't include `type`.
            return handler(v)

        # Import the symbol named in 'type'
        type_cls = mettagrid.util.module.load_symbol(t) if isinstance(t, str) else t

        if not inspect.isclass(type_cls):
            raise TypeError("'type' must point to a class")

        # `type_cls` can be more specific than `cls`.
        # This might matter when we load the config from YAML through a specific MapBuilderConfig subclass.
        # For example, `AsciiMapBuilder.Config.from_uri()` will return an instance of `AsciiMapBuilder.Config`.
        if not issubclass(type_cls, cls):
            raise TypeError(f"'type' {t} is not a subclass of {cls._type_str()}")

        data = {k: v for k, v in v.items() if k != "type"}
        result = type_cls.model_validate(data)

        assert isinstance(result, cls)  # should always be true because we checked the subclass relationship above

        return result


class WithMaxRetriesConfig(mettagrid.base_config.Config):
    max_retries: int = pydantic.Field(
        default=5,
        ge=0,
        description="Number of additional map samples to try when a builder raises ValueError during build().",
    )


AnyMapBuilderConfig = pydantic.SerializeAsAny[MapBuilderConfig]


ConfigT = typing.TypeVar("ConfigT", bound=MapBuilderConfig[typing.Any])


class MapBuilder(abc.ABC, typing.Generic[ConfigT]):
    """
    A base class for building MettaGridEnv game maps.

    Subclasses must:
    1. Inherit from MapBuilder[ConfigT], where ConfigT is a subclass of MapBuilderConfig.
    2. Define the build() method that returns a GameMap.
    """

    Config: type[ConfigT]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Look for MapBuilder base class
        try:
            base = next(base for base in getattr(cls, "__orig_bases__", ()) if typing.get_origin(base) is MapBuilder)
        except StopIteration:
            raise TypeError(
                f"{cls.__name__} must inherit from MapBuilder[…], with a concrete Config class parameter"
            ) from None

        # Set the Config class - this allows to use MapBuilder.Config shorthand
        config_cls = typing.get_args(base)[0]

        if config_cls._builder_cls:
            # Already bound to another MapBuilder class, so we need to clone it
            class CloneConfig(config_cls):  # type: ignore[misc]
                pass

            config_cls = CloneConfig

        config_cls._builder_cls = cls
        cls.Config = config_cls  # pyright: ignore[reportAttributeAccessIssue]
        return

    def __init__(self, config: ConfigT):
        self.config = config

    @abc.abstractmethod
    def build(self) -> GameMap: ...

    def build_for_num_agents(self, num_agents: int) -> GameMap:
        """
        Build a map and ensure it can accommodate the requested number of agents.

        Subclasses may override if they have a more efficient way to enforce spawn counts.
        """

        if isinstance(self.config, WithMaxRetriesConfig):
            retry_budget = self.config.max_retries
        else:
            retry_budget = 0

        for attempt in range(retry_budget + 1):
            try:
                game_map = self.build()
                self._designate_agent_spawn_points(game_map, num_agents)
                return game_map
            except ValueError as exc:
                if attempt == retry_budget:
                    raise exc
                logger.warning(
                    "Map build failed with ValueError on attempt %s/%s: %s; retrying",
                    attempt + 1,
                    retry_budget + 1,
                    exc,
                )
        raise ValueError(f"Failed to build map for {num_agents} agents")

    def shuffle_spawn_indices(self, indices: np.ndarray):
        """
        Shuffle the spawn indices. This method can be overridden to implement a
        different (seed-dependent) shuffle algorithm.
        """
        np.random.shuffle(indices)

    def _designate_agent_spawn_points(self, game_map: GameMap, num_agents: int) -> None:
        """
        Validate that the map provides enough spawn points and trim excess when necessary.
        """

        # Handle spawn points: treat them as potential spawn locations
        # If there are more spawn points than agents, replace the excess with empty spaces
        spawn_mask = np.char.startswith(game_map.grid, "agent")
        level_agents = np.count_nonzero(spawn_mask)

        if level_agents < num_agents:
            raise ValueError((f"Number of agents {num_agents} exceeds available spawn points {level_agents} in map."))

        if level_agents > num_agents:
            # Replace excess spawn points with empty spaces
            spawn_indices = np.argwhere(spawn_mask)
            # Randomly select num_agents spawn points to keep, replace the rest with empty
            self.shuffle_spawn_indices(spawn_indices)
            for idx in spawn_indices[num_agents:]:
                game_map.grid[tuple(idx)] = "empty"
