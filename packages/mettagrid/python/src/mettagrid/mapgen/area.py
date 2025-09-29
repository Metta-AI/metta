from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mettagrid.base_config import Config
from mettagrid.map_builder.map_builder import MapGrid


@dataclass
class Area:
    """
    A sub-area of the map grid.
    """

    # Full outer grid.
    # Useful when area is transformed or when the scene needs to be transplanted.
    # Scenes shouldn't use this field directly; instead, they should use the `self.grid` property.
    outer_grid: MapGrid

    # Absolute coordinates relative to the outer grid.
    x: int
    y: int

    width: int
    height: int

    tags: list[str] = field(default_factory=list)

    @property
    def grid(self) -> MapGrid:
        return self.outer_grid[self.y : self.y + self.height, self.x : self.x + self.width]

    @classmethod
    def root_area_from_grid(cls, grid: MapGrid) -> Area:
        return cls(outer_grid=grid, x=0, y=0, width=grid.shape[1], height=grid.shape[0])

    def as_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tags": self.tags,
        }

    def transplant_to_grid(self, grid: MapGrid, shift_x: int, shift_y: int, copy_grid: bool):
        original_grid = self.grid
        self.outer_grid = grid
        self.x += shift_x
        self.y += shift_y
        if copy_grid:
            self.grid[:] = original_grid


class AreaWhere(Config):
    tags: list[str] = []


class AreaQuery(Config):
    limit: int | None = None
    offset: int | None = None
    lock: str | None = None
    where: Literal["full"] | AreaWhere | None = None
    order_by: Literal["random", "first", "last"] = "random"
