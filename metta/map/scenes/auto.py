import numpy as np

from metta.common.util.config import Config
from metta.map.random.float import FloatDistribution
from metta.map.random.int import IntDistribution
from metta.map.scene import Scene
from metta.map.scenes.bsp import BSPLayout
from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.mirror import Mirror
from metta.map.scenes.random import Random
from metta.map.scenes.random_objects import RandomObjects
from metta.map.scenes.random_scene import RandomScene, RandomSceneCandidate
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere, ChildrenAction


class AutoParamsLayout(Config):
    grid: int
    bsp: int


class AutoParamsGrid(Config):
    rows: IntDistribution
    columns: IntDistribution


class AutoParamsBSP(Config):
    area_count: IntDistribution


class AutoParamsRoomSymmetry(Config):
    none: int
    horizontal: int
    vertical: int
    x4: int


class AutoParams(Config):
    num_agents: int = 0
    layout: AutoParamsLayout
    grid: AutoParamsGrid
    bsp: AutoParamsBSP
    room_symmetry: AutoParamsRoomSymmetry
    content: list[RandomSceneCandidate]
    objects: dict[str, FloatDistribution]
    room_objects: dict[str, FloatDistribution]


class Auto(Scene[AutoParams]):
    def get_children(self) -> list[ChildrenAction]:
        return [
            ChildrenAction(
                scene=AutoLayout.factory(self.params),
                where="full",
            ),
            ChildrenAction(
                scene=RandomObjects.factory({"object_ranges": self.params.objects}),
                where="full",
            ),
            ChildrenAction(
                scene=MakeConnected.factory({}),
                where="full",
            ),
            ChildrenAction(
                scene=Random.factory({"agents": self.params.num_agents}),
                where="full",
            ),
        ]

    def render(self):
        pass


class AutoLayout(Scene[AutoParams]):
    def get_children(self) -> list[ChildrenAction]:
        weights = np.array([self.params.layout.grid, self.params.layout.bsp], dtype=np.float32)
        weights /= weights.sum()
        layout = self.rng.choice(["grid", "bsp"], p=weights)

        def children_for_tag(tag: str) -> list[ChildrenAction]:
            return [
                ChildrenAction(
                    scene=AutoSymmetry.factory(self.params),
                    where=AreaWhere(tags=[tag]),
                ),
                ChildrenAction(
                    scene=RandomObjects.factory({"object_ranges": self.params.room_objects}),
                    where=AreaWhere(tags=[tag]),
                ),
            ]

        if layout == "grid":
            rows = self.params.grid.rows.sample(self.rng)
            columns = self.params.grid.columns.sample(self.rng)

            return [
                ChildrenAction(
                    scene=RoomGrid.factory(
                        {
                            "rows": rows,
                            "columns": columns,
                            "border_width": 0,  # randomize? probably not very useful
                        },
                        children=children_for_tag("room"),
                    ),
                    where="full",
                ),
            ]
        elif layout == "bsp":
            area_count = self.params.bsp.area_count.sample(self.rng)

            return [
                ChildrenAction(
                    scene=BSPLayout.factory(
                        {"area_count": area_count},
                        children=children_for_tag("zone"),
                    ),
                    where="full",
                ),
            ]
        else:
            raise ValueError(f"Invalid layout: {layout}")

    def render(self):
        pass


class AutoSymmetry(Scene[AutoParams]):
    def get_children(self) -> list[ChildrenAction]:
        weights = np.array(
            [
                self.params.room_symmetry.none,
                self.params.room_symmetry.horizontal,
                self.params.room_symmetry.vertical,
                self.params.room_symmetry.x4,
            ],
            dtype=np.float32,
        )
        weights /= weights.sum()
        symmetry = self.rng.choice(["none", "horizontal", "vertical", "x4"], p=weights)

        scene = RandomScene.factory({"candidates": self.params.content})
        if symmetry != "none":
            scene = Mirror.factory({"scene": scene, "symmetry": symmetry})

        return [ChildrenAction(scene=scene, where="full")]

    def render(self):
        pass
