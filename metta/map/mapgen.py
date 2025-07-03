from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.map.scene import load_class, make_scene, scene_cfg_to_dict
from metta.map.scenes.room_grid import RoomGrid, RoomGridParams
from metta.map.types import MapGrid
from metta.mettagrid.level_builder import Level, LevelBuilder

from .types import Area, AreaWhere, ChildrenAction, SceneCfg


# Root map generator, based on scenes.
@dataclass
class MapGen(LevelBuilder):
    # Root scene configuration.
    # In YAML configs, this is usually the dict with `type` and `params` keys, and possible children.
    # This is the only required parameter.
    root: SceneCfg

    # Inner grid size. Doesn't take outer border into account.
    # If `instances` is set, this is the size used for each instance.
    # If width and height are not set, the root scene must provide an intrinsic size.
    width: int | None = None
    height: int | None = None

    # Default border_width value guarantees that agents don't see beyond the outer walls.
    # This value usually shouldn't be changed.
    border_width: int = 5

    # Number of root scene instances to generate.
    # If set, the map will be generated as a grid of instances, with the given border width.
    # This is useful for additional parallelization.
    # By default, the map will be generated as a single root scene instance, with the given width and height.
    instances: int = 1
    instance_border_width: int = 5

    def build(self):
        instance_rows = int(np.ceil(np.sqrt(self.instances)))
        instance_cols = int(np.ceil(self.instances / instance_rows))

        if not self.width or not self.height:
            dict_cfg = scene_cfg_to_dict(self.root)
            root_cls = load_class(dict_cfg["type"])
            intrinsic_size = root_cls.intrinsic_size(dict_cfg.get("params", {}))
            if not intrinsic_size:
                raise ValueError("width and height must be provided if the root scene has no intrinsic size")
            self.height, self.width = intrinsic_size

        self.inner_width = self.width * instance_cols + (instance_cols - 1) * self.instance_border_width
        self.inner_height = self.height * instance_rows + (instance_rows - 1) * self.instance_border_width

        if isinstance(self.root, DictConfig):
            self.root = OmegaConf.to_container(self.root)  # type: ignore

        root_scene_cfg = self.root

        if self.instances > 1:
            root_scene_cfg = RoomGrid.factory(
                RoomGridParams(
                    rows=instance_rows,
                    columns=instance_cols,
                    border_width=self.instance_border_width,
                ),
                children=[
                    ChildrenAction(
                        scene=self.root,
                        where=AreaWhere(tags=["room"]),
                    )
                ],
            )

        bw = self.border_width

        self.grid: MapGrid = np.full(
            (self.inner_height + 2 * bw, self.inner_width + 2 * bw),
            "empty",
            dtype="<U50",
        )

        # draw outer walls
        # note that the inner walls when instances > 1 will be drawn by the RoomGrid scene
        self.grid[:bw, :] = "wall"
        self.grid[-bw:, :] = "wall"
        self.grid[:, :bw] = "wall"
        self.grid[:, -bw:] = "wall"

        inner_grid = self.grid[
            bw : bw + self.inner_height,
            bw : bw + self.inner_width,
        ]

        inner_area = Area(x=bw, y=bw, width=self.inner_width, height=self.inner_height, grid=inner_grid, tags=[])

        self.root_scene = make_scene(root_scene_cfg, inner_area, rng=np.random.default_rng())

        self.root_scene.render_with_children()
        # TODO: support labels, similarly to `mettagrid.room.room.Room`
        return Level(self.grid, [])

    def get_scene_tree(self) -> dict:
        return self.root_scene.get_scene_tree()
