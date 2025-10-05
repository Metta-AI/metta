import numpy as np

from mettagrid.base_config import Config
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.random.float import FloatDistribution
from mettagrid.mapgen.random.int import IntDistribution
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.bsp import BSPLayout
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.mirror import Mirror
from mettagrid.mapgen.scenes.random import Random
from mettagrid.mapgen.scenes.random_objects import RandomObjects
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate
from mettagrid.mapgen.scenes.room_grid import RoomGrid


class AutoConfigLayout(Config):
    grid: int
    bsp: int


class AutoConfigGrid(Config):
    rows: IntDistribution
    columns: IntDistribution


class AutoConfigBSP(Config):
    area_count: IntDistribution


class AutoConfigRoomSymmetry(Config):
    none: int
    horizontal: int
    vertical: int
    x4: int


class AutoConfig(SceneConfig):
    num_agents: int = 0
    layout: AutoConfigLayout
    grid: AutoConfigGrid
    bsp: AutoConfigBSP
    room_symmetry: AutoConfigRoomSymmetry
    content: list[RandomSceneCandidate]
    objects: dict[str, FloatDistribution]
    room_objects: dict[str, FloatDistribution]


class Auto(Scene[AutoConfig]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=AutoLayout.Config(auto_config=self.config),
                where="full",
            ),
            ChildrenAction(
                scene=RandomObjects.Config(object_ranges=self.config.objects),
                where="full",
            ),
            ChildrenAction(
                scene=MakeConnected.Config(),
                where="full",
            ),
            ChildrenAction(
                scene=Random.Config(agents=self.config.num_agents),
                where="full",
            ),
        ]

    def render(self):
        pass


class AutoLayoutConfig(SceneConfig):
    auto_config: AutoConfig


class AutoLayout(Scene[AutoLayoutConfig]):
    def get_children(self) -> list[ChildrenAction]:
        weights = np.array([self.config.auto_config.layout.grid, self.config.auto_config.layout.bsp], dtype=np.float32)
        weights /= weights.sum()
        layout = self.rng.choice(["grid", "bsp"], p=weights)

        def children_actions_for_tag(tag: str) -> list[ChildrenAction]:
            return [
                ChildrenAction(
                    scene=AutoSymmetry.Config(auto_config=self.config.auto_config),
                    where=AreaWhere(tags=[tag]),
                ),
                ChildrenAction(
                    scene=RandomObjects.Config(object_ranges=self.config.auto_config.room_objects),
                    where=AreaWhere(tags=[tag]),
                ),
            ]

        if layout == "grid":
            rows = self.config.auto_config.grid.rows.sample(self.rng)
            columns = self.config.auto_config.grid.columns.sample(self.rng)

            return [
                ChildrenAction(
                    scene=RoomGrid.Config(
                        rows=rows,
                        columns=columns,
                        border_width=0,  # randomize? probably not very useful
                        children=children_actions_for_tag("room"),
                    ),
                    where="full",
                ),
            ]
        elif layout == "bsp":
            area_count = self.config.auto_config.bsp.area_count.sample(self.rng)

            return [
                ChildrenAction(
                    scene=BSPLayout.Config(
                        area_count=area_count,
                        children=children_actions_for_tag("zone"),
                    ),
                    where="full",
                ),
            ]
        else:
            raise ValueError(f"Invalid layout: {layout}")

    def render(self):
        pass


class AutoSymmetryConfig(SceneConfig):
    auto_config: AutoConfig


class AutoSymmetry(Scene[AutoSymmetryConfig]):
    def get_children(self) -> list[ChildrenAction]:
        weights = np.array(
            [
                self.config.auto_config.room_symmetry.none,
                self.config.auto_config.room_symmetry.horizontal,
                self.config.auto_config.room_symmetry.vertical,
                self.config.auto_config.room_symmetry.x4,
            ],
            dtype=np.float32,
        )
        weights /= weights.sum()
        symmetry = self.rng.choice(["none", "horizontal", "vertical", "x4"], p=weights)

        scene = RandomScene.Config(candidates=self.config.auto_config.content)
        if symmetry != "none":
            scene = Mirror.Config(scene=scene, symmetry=symmetry)

        return [ChildrenAction(scene=scene, where="full")]

    def render(self):
        pass
