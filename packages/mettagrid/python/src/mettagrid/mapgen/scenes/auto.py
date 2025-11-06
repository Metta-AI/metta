import numpy as np

import mettagrid.base_config
import mettagrid.mapgen.area
import mettagrid.mapgen.random.float
import mettagrid.mapgen.random.int
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.bsp
import mettagrid.mapgen.scenes.make_connected
import mettagrid.mapgen.scenes.mirror
import mettagrid.mapgen.scenes.random
import mettagrid.mapgen.scenes.random_objects
import mettagrid.mapgen.scenes.random_scene
import mettagrid.mapgen.scenes.room_grid


class AutoConfigLayout(mettagrid.base_config.Config):
    grid: int
    bsp: int


class AutoConfigGrid(mettagrid.base_config.Config):
    rows: mettagrid.mapgen.random.int.IntDistribution
    columns: mettagrid.mapgen.random.int.IntDistribution


class AutoConfigBSP(mettagrid.base_config.Config):
    area_count: mettagrid.mapgen.random.int.IntDistribution


class AutoConfigRoomSymmetry(mettagrid.base_config.Config):
    none: int
    horizontal: int
    vertical: int
    x4: int


class AutoConfig(mettagrid.mapgen.scene.SceneConfig):
    num_agents: int = 0
    layout: AutoConfigLayout
    grid: AutoConfigGrid
    bsp: AutoConfigBSP
    room_symmetry: AutoConfigRoomSymmetry
    content: list[mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate]
    objects: dict[str, mettagrid.mapgen.random.float.FloatDistribution]
    room_objects: dict[str, mettagrid.mapgen.random.float.FloatDistribution]


class Auto(mettagrid.mapgen.scene.Scene[AutoConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=AutoLayout.Config(auto_config=self.config),
                where="full",
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.random_objects.RandomObjects.Config(object_ranges=self.config.objects),
                where="full",
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.make_connected.MakeConnected.Config(),
                where="full",
            ),
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.random.Random.Config(agents=self.config.num_agents),
                where="full",
            ),
        ]

    def render(self):
        pass


class AutoLayoutConfig(mettagrid.mapgen.scene.SceneConfig):
    auto_config: AutoConfig


class AutoLayout(mettagrid.mapgen.scene.Scene[AutoLayoutConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        weights = np.array([self.config.auto_config.layout.grid, self.config.auto_config.layout.bsp], dtype=np.float32)
        weights /= weights.sum()
        layout = self.rng.choice(["grid", "bsp"], p=weights)

        def children_actions_for_tag(tag: str) -> list[mettagrid.mapgen.scene.ChildrenAction]:
            return [
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=AutoSymmetry.Config(auto_config=self.config.auto_config),
                    where=mettagrid.mapgen.area.AreaWhere(tags=[tag]),
                ),
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.random_objects.RandomObjects.Config(
                        object_ranges=self.config.auto_config.room_objects
                    ),
                    where=mettagrid.mapgen.area.AreaWhere(tags=[tag]),
                ),
            ]

        if layout == "grid":
            rows = self.config.auto_config.grid.rows.sample(self.rng)
            columns = self.config.auto_config.grid.columns.sample(self.rng)

            return [
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.room_grid.RoomGrid.Config(
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
                mettagrid.mapgen.scene.ChildrenAction(
                    scene=mettagrid.mapgen.scenes.bsp.BSPLayout.Config(
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


class AutoSymmetryConfig(mettagrid.mapgen.scene.SceneConfig):
    auto_config: AutoConfig


class AutoSymmetry(mettagrid.mapgen.scene.Scene[AutoSymmetryConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
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

        scene = mettagrid.mapgen.scenes.random_scene.RandomScene.Config(candidates=self.config.auto_config.content)
        if symmetry != "none":
            scene = mettagrid.mapgen.scenes.mirror.Mirror.Config(scene=scene, symmetry=symmetry)

        return [mettagrid.mapgen.scene.ChildrenAction(scene=scene, where="full")]

    def render(self):
        pass
