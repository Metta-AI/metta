from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import Field

from metta.common.util.config import Config

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

    def __init__(self, grid: MapGrid):
        self.grid = grid


class MapBuilderConfig(Config):
    """
    Configuration for building a game map.
    """

    # Discriminator field for polymorphic serialization
    type: str = Field(..., description="Map builder type discriminator")

    @abstractmethod
    def create(self) -> "MapBuilder": ...


class MapBuilder(ABC):
    """
    A base class for building MettaGridEnv game maps.
    """

    def __init__(self, config: MapBuilderConfig):
        self._config = config

    @abstractmethod
    def build(self) -> GameMap: ...


# MapBuilderConfigUnion is defined in __init__.py to avoid circular imports
