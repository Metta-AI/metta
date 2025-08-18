from typing import Callable

from metta.map.mapgen import MapGen
from metta.map.random.int import IntConstantDistribution
from metta.map.scene import ChildrenAction
from metta.map.scenes.ascii import Ascii
from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.layout import Layout, LayoutArea
from metta.map.scenes.maze import Maze
from metta.map.scenes.mean_distance import MeanDistance
from metta.map.scenes.radial_maze import RadialMaze
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere
from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_ascii_env(max_steps: int, ascii_map: str, border_width: int = 1) -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config.with_ascii(
        ascii_map, border_width=border_width
    )
    return env


def make_corridors_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=450,
        ascii_map="configs/env/mettagrid/maps/navigation/corridors.map",
    )


def make_cylinder_easy_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=250,
        ascii_map="configs/env/mettagrid/maps/navigation/cylinder_easy.map",
    )


def make_cylinder_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=250,
        border_width=10,  # I have no idea if this is important -- Slava
        ascii_map="configs/env/mettagrid/maps/navigation/cylinder.map",
    )


def make_honeypot_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="configs/env/mettagrid/maps/navigation/honeypot.map",
    )


def make_knotty_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=500,
        ascii_map="configs/env/mettagrid/maps/navigation/knotty.map",
    )


def make_memory_palace_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=200,
        ascii_map="configs/env/mettagrid/maps/navigation/memory_palace.map",
    )


def make_obstacles0_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=100,
        ascii_map="configs/env/mettagrid/maps/navigation/obstacles0.map",
    )


def make_obstacles1_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 300
    # Make a copy of the altar config before modifying
    env.game.objects["altar"] = env.game.objects["altar"].model_copy()
    env.game.objects["altar"].cooldown = 255
    env.game.map_builder = MapGen.Config(
        border_width=1,
        root=Ascii.factory(
            params=Ascii.Params(
                uri="configs/env/mettagrid/maps/navigation/obstacles1.map",
            ),
        ),
    )
    return env


def make_obstacles2_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=350,
        ascii_map="configs/env/mettagrid/maps/navigation/obstacles2.map",
    )


def make_obstacles3_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="configs/env/mettagrid/maps/navigation/obstacles3.map",
    )


def make_radial_large_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=1000,
        ascii_map="configs/env/mettagrid/maps/navigation/radial_large.map",
    )


def make_radial_mini_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=150,
        ascii_map="configs/env/mettagrid/maps/navigation/radial_mini.map",
    )


def make_radial_small_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=120,
        ascii_map="configs/env/mettagrid/maps/navigation/radial_small.map",
    )


def make_radialmaze_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 200
    env.game.map_builder = MapGen.Config(
        width=20,
        height=20,
        border_width=1,
        root=RadialMaze.factory(
            params=RadialMaze.Params(
                arms=4,
                arm_length=8,
            ),
            children_actions=[
                # put agent at the center
                ChildrenAction(
                    where=AreaWhere(tags=["center"]),
                    scene=InlineAscii.factory(
                        params=InlineAscii.Params(data="@"),
                    ),
                ),
                # put altars at the first 3 endpoints
                ChildrenAction(
                    where=AreaWhere(tags=["endpoint"]),
                    limit=3,
                    order_by="first",
                    scene=InlineAscii.factory(
                        params=InlineAscii.Params(data="_"),
                    ),
                ),
            ],
        ),
    )
    return env


def make_swirls_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=350,
        ascii_map="configs/env/mettagrid/maps/navigation/swirls.map",
    )


def make_thecube_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=350,
        ascii_map="configs/env/mettagrid/maps/navigation/thecube.map",
    )


def make_walkaround_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=250,
        ascii_map="configs/env/mettagrid/maps/navigation/walkaround.map",
    )


def make_walls_outofsight_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 120
    env.game.map_builder = MapGen.Config(
        width=25,
        height=25,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=15,
                objects={"altar": 3, "wall": 12},
            )
        ),
    )
    return env


def make_walls_sparse_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 250
    env.game.map_builder = MapGen.Config(
        width=35,
        height=35,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=25,
                objects={"altar": 3, "wall": 12},
            )
        ),
    )
    return env


def make_walls_withinsight_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 75
    env.game.map_builder = MapGen.Config(
        width=12,
        height=12,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=10,
                objects={"altar": 3, "wall": 4},
            )
        ),
    )
    return env


def make_wanderout_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=500,
        ascii_map="configs/env/mettagrid/maps/navigation/wanderout.map",
    )


def make_emptyspace_outofsight_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 150
    env.game.map_builder = MapGen.Config(
        width=25,
        height=25,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=15,
                objects={"altar": 3},
            )
        ),
    )
    return env


def make_emptyspace_withinsight_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 45
    env.game.map_builder = MapGen.Config(
        width=12,
        height=12,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=10,
                objects={"altar": 3},
            )
        ),
    )
    return env


def make_emptyspace_sparse_medium_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 1000
    env.game.map_builder = MapGen.Config(
        width=100,
        height=100,
        border_width=5,
        root=RoomGrid.factory(
            params=RoomGrid.Params(
                border_width=0,
                layout=[
                    ["border", "border", "border", "border", "border"],
                    ["border", "middle", "middle", "middle", "border"],
                    ["border", "middle", "middle", "middle", "border"],
                    ["border", "middle", "middle", "middle", "border"],
                    ["border", "border", "border", "border", "border"],
                ],
            ),
            children_actions=[
                ChildrenAction(
                    where=AreaWhere(tags=["middle"]),
                    limit=3,
                    lock="lock",
                    scene=Random.factory(
                        params=Random.Params(objects={"altar": 1}),
                    ),
                ),
                ChildrenAction(
                    where=AreaWhere(tags=["middle"]),
                    limit=1,
                    lock="lock",
                    scene=Random.factory(
                        params=Random.Params(agents=1),
                    ),
                ),
            ],
        ),
    )
    return env


def make_emptyspace_sparse_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 300
    env.game.map_builder = MapGen.Config(
        width=60,
        height=60,
        border_width=3,
        root=MeanDistance.factory(
            params=MeanDistance.Params(
                mean_distance=30,
                objects={"altar": 3},
            )
        ),
    )
    return env


def make_labyrinth_env() -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = 250
    env.game.map_builder = MapGen.Config(
        width=31,
        height=31,
        border_width=2,
        root=Maze.factory(
            params=Maze.Params(
                algorithm="dfs",
                room_size=IntConstantDistribution(value=3),
                wall_size=IntConstantDistribution(value=1),
            ),
            children_actions=[
                # agent in the top-left corner
                ChildrenAction(
                    where=AreaWhere(tags=["top-left"]),
                    scene=InlineAscii.factory(
                        params=InlineAscii.Params(
                            data="@",
                            # room center (for 3x3 rooms)
                            row=1,
                            column=1,
                        ),
                    ),
                ),
                # three altars in the center
                ChildrenAction(
                    where="full",
                    scene=Layout.factory(
                        params=Layout.Params(
                            areas=[
                                LayoutArea(
                                    width=7, height=5, tag="reward", placement="center"
                                ),
                            ],
                        ),
                        children_actions=[
                            ChildrenAction(
                                where=AreaWhere(tags=["reward"]),
                                scene=InlineAscii.factory(
                                    params=InlineAscii.Params(
                                        data="""
                                            .......
                                            .......
                                            ..___..
                                            .......
                                            .......

                                        """
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        ),
    )
    return env


def make_navigation_eval_suite() -> list[SimulationConfig]:
    env_config_makers: list[Callable[[], EnvConfig]] = [
        make_corridors_env,
        make_cylinder_easy_env,
        make_cylinder_env,
        make_honeypot_env,
        make_knotty_env,
        make_memory_palace_env,
        make_obstacles0_env,
        make_obstacles1_env,
        make_obstacles2_env,
        make_obstacles3_env,
        make_radial_large_env,
        make_radial_mini_env,
        make_radial_small_env,
        make_radialmaze_env,
        make_swirls_env,
        make_thecube_env,
        make_walkaround_env,
        make_walls_outofsight_env,
        make_walls_sparse_env,
        make_walls_withinsight_env,
        make_wanderout_env,
        make_emptyspace_outofsight_env,
        make_emptyspace_withinsight_env,
        make_emptyspace_sparse_medium_env,
        make_emptyspace_sparse_env,
        make_labyrinth_env,
    ]

    def fn_to_sim_name(fn: Callable[[], EnvConfig]) -> str:
        return fn.__name__.replace("make_", "").replace("_env", "")

    return [
        SimulationConfig(
            name=fn_to_sim_name(env_config_maker),
            env=env_config_maker(),
        )
        for env_config_maker in env_config_makers
    ]
