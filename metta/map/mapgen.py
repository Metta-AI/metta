from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.map.scene import make_scene
from metta.map.types import MapGrid
from metta.mettagrid.level_builder import Level, LevelBuilder

from .types import Area, SceneCfg


# Root map generator, based on scenes.
@dataclass
class MapGen(LevelBuilder):
    width: int
    height: int
    root: SceneCfg
    # Default value guarantees that agents don't see beyond the outer walls.
    # Usually shouldn't be changed.
    border_width: int = 5

    def __post_init__(self):
        super().__init__()

        self.grid: MapGrid = np.full(
            (self.height + 2 * self.border_width, self.width + 2 * self.border_width), "empty", dtype="<U50"
        )
        self.grid[: self.border_width, :] = "wall"
        self.grid[-self.border_width :, :] = "wall"
        self.grid[:, : self.border_width] = "wall"
        self.grid[:, -self.border_width :] = "wall"

        if isinstance(self.root, DictConfig):
            self.root = OmegaConf.to_container(self.root)  # type: ignore

        self.root_scene = make_scene(self.root, self.inner_area(), rng=np.random.default_rng())

    def inner_area(self) -> Area:
        x = self.border_width
        y = self.border_width
        grid = self.grid[y : y + self.height, x : x + self.width]

        return Area(x=x, y=y, width=self.width, height=self.height, grid=grid, tags=[])

    def build(self):
        self.root_scene.render_with_children()
        # TODO: support labels, similarly to `mettagrid.room.room.Room`
        return Level(self.grid, [])

    def get_scene_tree(self) -> dict:
        return self.root_scene.get_scene_tree()
