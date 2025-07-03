from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.map.scene import make_scene
from metta.map.scenes.room_grid import RoomGrid, RoomGridParams
from metta.map.types import MapGrid
from metta.mettagrid.level_builder import Level, LevelBuilder

from .types import Area, ChildrenAction, SceneCfg


# Root map generator, based on scenes.
@dataclass
class MapGen(LevelBuilder):
    width: int
    height: int
    root: SceneCfg
    # Default value guarantees that agents don't see beyond the outer walls.
    # Usually shouldn't be changed.
    border_width: int = 5
    num_rooms: int = 1
    room_border_width: int = 5

    def __post_init__(self):
        super().__init__()

        room_rows = int(np.ceil(np.sqrt(self.num_rooms)))
        room_cols = int(np.ceil(self.num_rooms / room_rows))

        full_width = self.width + 2 * self.border_width + (room_cols - 1) * self.room_border_width
        full_height = self.height + 2 * self.border_width + (room_rows - 1) * self.room_border_width

        if isinstance(self.root, DictConfig):
            self.root = OmegaConf.to_container(self.root)  # type: ignore

        root_scene_cfg = self.root

        if self.num_rooms > 1:
            root_scene_cfg = RoomGrid.factory(
                RoomGridParams(
                    rows=room_rows,
                    columns=room_cols,
                    border_width=self.room_border_width,
                ),
                children=[
                    ChildrenAction(
                        scene=self.root,
                        where="full",
                    )
                ],
            )

        self.grid: MapGrid = np.full((full_height, full_width), "empty", dtype="<U50")

        # draw outer walls
        # note that the inner walls when num_rooms > 1 will be drawn by the RoomGrid scene
        self.grid[: self.border_width, :] = "wall"
        self.grid[-self.border_width :, :] = "wall"
        self.grid[:, : self.border_width] = "wall"
        self.grid[:, -self.border_width :] = "wall"

        self.root_scene = make_scene(root_scene_cfg, self.inner_area(), rng=np.random.default_rng())

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
