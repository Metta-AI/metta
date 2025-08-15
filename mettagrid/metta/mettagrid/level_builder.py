from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

MapGrid: TypeAlias = npt.NDArray[np.str_]


map_grid_dtype = np.dtype("<U20")


def create_grid(height: int, width: int, fill_value: str = "empty") -> MapGrid:
    """
    Creates a NumPy grid with the given height and width, pre-filled with the specified fill_value.
    """
    return np.full((height, width), fill_value, dtype=map_grid_dtype)


@dataclass
class Level:
    """
    Represents a level in the MettaGrid game.

    Note: this is intentionally called "Level" instead of "Map" because `map` is a reserved word in Python.
    """

    # Two-dimensional grid of strings.
    # Possible values: "wall", "empty", "agent", etc.
    # For the full list, see `mettagrid_c.cpp`.
    grid: MapGrid

    # List of labels. These will be used for `rewards/map:...` episode stats.
    labels: list[str]


class LevelBuilder(ABC):
    """
    A base class for building MettaGridEnv levels.

    Right now we have two implementations:
    1. `mettagrid.room.room.Room` and its subclasses
    2. `metta.map.mapgen.MapGen`

    MapGen system is more flexible and the current plan is to refactor away the Room class hierarchy.
    """

    @abstractmethod
    def build(self) -> Level: ...
