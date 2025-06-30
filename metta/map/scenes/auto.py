import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.scenes.bsp import BSPLayout
from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.mirror import Mirror
from metta.map.scenes.random import Random
from metta.map.scenes.random_objects import RandomObjects
from metta.map.scenes.random_scene import RandomScene, RandomSceneCandidate
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import Area, AreaWhere, ChildrenAction
from metta.map.utils.random import FloatDistribution, IntDistribution, sample_int_distribution


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
                scene=lambda area: AutoLayout(area=area, params=self.params, seed=self.rng),
                where="full",
            ),
            ChildrenAction(
                scene=lambda area: RandomObjects(
                    area=area, params={"object_ranges": self.params.objects}, seed=self.rng
                ),
                where="full",
            ),
            ChildrenAction(
                scene=lambda area: MakeConnected(area=area, seed=self.rng),
                where="full",
            ),
            ChildrenAction(
                scene=lambda area: Random(area=area, params={"agents": self.params.num_agents}, seed=self.rng),
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
                    scene=lambda area: AutoSymmetry(area=area, params=self.params, seed=self.rng),
                    where=AreaWhere(tags=[tag]),
                ),
                ChildrenAction(
                    scene=lambda area: RandomObjects(
                        area=area, params={"object_ranges": self.params.room_objects}, seed=self.rng
                    ),
                    where=AreaWhere(tags=[tag]),
                ),
            ]

        if layout == "grid":
            rows = sample_int_distribution(self.params.grid.rows, self.rng)
            columns = sample_int_distribution(self.params.grid.columns, self.rng)

            return [
                ChildrenAction(
                    scene=lambda area: RoomGrid(
                        area=area,
                        params={
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
            area_count = sample_int_distribution(self.params.bsp.area_count, self.rng)

            return [
                ChildrenAction(
                    scene=lambda area: BSPLayout(
                        area=area,
                        params={"area_count": area_count},
                        children=children_for_tag("zone"),
                        seed=self.rng,
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

        def get_random_scene(area: Area) -> Scene:
            return RandomScene(area=area, params={"candidates": self.params.content}, seed=self.rng)

        def get_scene(area: Area) -> Scene:
            if symmetry == "none":
                return get_random_scene(area)
            else:
                return Mirror(area=area, params={"scene": get_random_scene, "symmetry": symmetry}, seed=self.rng)

        return [ChildrenAction(scene=get_scene, where="full")]

    def render(self):
        pass
