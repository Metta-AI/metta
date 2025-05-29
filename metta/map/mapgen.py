from dataclasses import dataclass

import numpy as np

from metta.map.types import MapGrid
from mettagrid.level_builder import Level, LevelBuilder

from .scene import SceneCfg, make_scene


# Root map generator, based on nodes.
@dataclass
class MapGen(LevelBuilder):
    width: int
    height: int
    root: SceneCfg
    border_width: int = 1

    def __post_init__(self):
        super().__init__()
        self.root_scene = make_scene(self.root)

        self.grid: MapGrid = np.full(
            (self.height + 2 * self.border_width, self.width + 2 * self.border_width), "empty", dtype="<U50"
        )
        self.grid[: self.border_width, :] = "wall"
        self.grid[-self.border_width :, :] = "wall"
        self.grid[:, : self.border_width] = "wall"
        self.grid[:, -self.border_width :] = "wall"

    def inner_grid(self) -> MapGrid:
        if self.border_width > 0:
            return self.grid[
                self.border_width : -self.border_width,
                self.border_width : -self.border_width,
            ]
        else:
            return self.grid

    def build(self):
        root_node = self.root_scene.make_node(self.inner_grid())

        root_node.render()
        # TODO: support labels, similarly to `mettagrid.room.room.Room`
        return Level(self.grid, [])
