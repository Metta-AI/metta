import numpy as np
from experiments.recipes.arena import make_mettagrid
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import MapGen
from mettagrid.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.bsp import BSP
from mettagrid.mapgen.scenes.random import Random
from mettagrid.mapgen.types import AreaWhere


class RandomWithCenterBiasParams(Config):
    pass


class RandomWithCenterBias(Scene[RandomWithCenterBiasParams]):
    def get_children(self) -> list[ChildrenAction]:
        area = self.area
        abs_width = area.abs_grid.shape[1]
        abs_height = area.abs_grid.shape[0]
        abs_center_x = abs_width / 2
        abs_center_y = abs_height / 2
        area_center_x = area.x + area.width / 2
        area_center_y = area.y + area.height / 2
        distance_from_center = np.sqrt(
            (area_center_x - abs_center_x) ** 2 + (area_center_y - abs_center_y) ** 2
        )
        max_distance_from_center = np.sqrt((abs_center_x) ** 2 + (abs_center_y) ** 2)

        ratio_from_center = distance_from_center / max_distance_from_center

        return [
            ChildrenAction(
                scene=Random.factory(
                    Random.Params(
                        objects={
                            "mine_red": int(ratio_from_center * 5),
                        },
                    )
                ),
                where="full",
            ),
        ]

    def render(self):
        pass


def bsp_like() -> MettaGridConfig:
    config = make_mettagrid()
    config.game.map_builder = MapGen.Config(
        instances=1,
        border_width=1,
        width=120,
        height=120,
        root=BSP.factory(
            params=BSP.Params(
                rooms=30,
                min_room_size=3,
                min_room_size_ratio=0.9,
                max_room_size_ratio=0.9,
            ),
            children_actions=[
                ChildrenAction(
                    scene=Random.factory(Random.Params(agents=4)),
                    where=AreaWhere(tags=["room"]),
                    limit=1,
                    lock="rooms",
                ),
                ChildrenAction(
                    scene=RandomWithCenterBias.factory(RandomWithCenterBias.Params()),
                    where=AreaWhere(tags=["room"]),
                    lock="rooms",
                    order_by="first",
                    limit=3,
                ),
            ],
        ),
    )
    return config
