"""Map builder module for MettaGrid."""

from typing import Annotated, Union

from pydantic import Field

from metta.map.mapgen import MapGenConfig

from .ascii import AsciiMapBuilderConfig
from .map_builder import GameMap, MapBuilder, MapBuilderConfig
from .maze import MazeKruskalMapBuilderConfig, MazePrimMapBuilderConfig
from .random import RandomMapBuilderConfig

# Define the discriminated union for core map builders only
# MapGenConfig and TerrainFromNumpyConfig are in a separate union in metta.map
MapBuilderConfigUnion = Annotated[
    Union[
        Annotated[AsciiMapBuilderConfig, Field(discriminator="type")],
        Annotated[MazePrimMapBuilderConfig, Field(discriminator="type")],
        Annotated[MazeKruskalMapBuilderConfig, Field(discriminator="type")],
        Annotated[RandomMapBuilderConfig, Field(discriminator="type")],
        # FIXME #dehydration - mettagrid shouldn't import from metta
        Annotated[MapGenConfig, Field(discriminator="type")],
    ],
    Field(discriminator="type"),
]

__all__ = [
    "GameMap",
    "MapBuilder",
    "MapBuilderConfig",
    "MapBuilderConfigUnion",
    "AsciiMapBuilderConfig",
    "MazePrimMapBuilderConfig",
    "MazeKruskalMapBuilderConfig",
    "RandomMapBuilderConfig",
]
