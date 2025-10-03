from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from mettagrid.base_config import Config

# We store maps as 2D arrays of object names.
# "empty" means an empty cell; "wall" means a wall, etc.
MapGrid: TypeAlias = npt.NDArray[np.str_]
map_grid_dtype = np.dtype("<U20")


@dataclass
class Area:
    # absolute coordinates
    x: int
    y: int

    width: int
    height: int

    # slice of the outer grid (must match `width` and `height`)
    grid: MapGrid

    tags: list[str]

    @classmethod
    def root_area_from_grid(cls, grid: MapGrid) -> "Area":
        return cls(x=0, y=0, width=grid.shape[1], height=grid.shape[0], grid=grid, tags=[])

    def as_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tags": self.tags,
        }

    def transplant_to_grid(self, grid: MapGrid, shift_x: int, shift_y: int):
        self.x += shift_x
        self.y += shift_y
        self.grid = grid[self.y : self.y + self.height, self.x : self.x + self.width]

    def __getitem__(self, key) -> "Area":
        # TODO - I think this method doesn't support negative indices.
        # (written by Claude)

        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_slice = key
        elif isinstance(key, (int, slice)):
            row_slice = key
            col_slice = slice(None)
        else:
            raise TypeError("Area indices must be integers or slices")

        # Convert integers to slice objects
        if isinstance(row_slice, int):
            row_slice = slice(row_slice, row_slice + 1)
        if isinstance(col_slice, int):
            col_slice = slice(col_slice, col_slice + 1)

        # Get the sliced grid
        sliced_grid = self.grid[row_slice, col_slice]

        # Calculate new coordinates
        row_start = row_slice.start if row_slice.start is not None else 0
        col_start = col_slice.start if col_slice.start is not None else 0

        new_x = self.x + col_start
        new_y = self.y + row_start
        new_height, new_width = sliced_grid.shape

        return Area(x=new_x, y=new_y, width=new_width, height=new_height, grid=sliced_grid, tags=self.tags.copy())


class AreaWhere(Config):
    tags: list[str] = []


class AreaQuery(Config):
    limit: int | None = None
    offset: int | None = None
    lock: str | None = None
    where: Literal["full"] | AreaWhere | None = None
    order_by: Literal["random", "first", "last"] = "random"
