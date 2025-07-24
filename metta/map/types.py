from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from metta.common.util.config import Config

# We store maps as 2D arrays of object names.
# "empty" means an empty cell; "wall" means a wall, etc. See `metta.mettagrid.char_encoder` for the full list.
#
# Properly shaped version, `np.ndarray[tuple[int, int], np.dtype[np.str_]]`,
# would be better, but slices from numpy arrays are not typed properly, which makes it too annoying to use.
MapGrid: TypeAlias = npt.NDArray[np.str_]


@dataclass
class Area:
    # absolute coordinates
    x: int
    y: int

    width: int
    height: int

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


# Scene configs can be either:
# - a dict with `type`, `params`, and optionally `children` keys (this is how we define scenes in YAML configs)
# - a string path to a scene config file (this is how we load reusable scene configs from `scenes/` directory)
# - a function that takes a MapGrid and returns a Scene instance (useful for children actions produced in Python code)
#
# See `metta.map.scene.make_scene` implementation for more details.
SceneCfg = dict | str | Callable[[Area, np.random.Generator], Any]


class AreaWhere(Config):
    tags: list[str] = []


class AreaQuery(Config):
    limit: int | None = None
    offset: int | None = None
    lock: str | None = None
    where: Literal["full"] | AreaWhere | None = None
    order_by: Literal["random", "first", "last"] = "random"


class ChildrenAction(AreaQuery):
    scene: SceneCfg
