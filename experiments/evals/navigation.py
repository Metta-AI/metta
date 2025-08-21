from metta.map.mapgen import MapGen
from metta.map.random.int import IntConstantDistribution
from metta.map.scene import ChildrenAction
from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.scenes.layout import Layout, LayoutArea
from metta.map.scenes.maze import Maze
from metta.map.scenes.mean_distance import MeanDistance
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere
from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_ascii_env(max_steps: int, ascii_map: str, border_width: int = 1) -> EnvConfig:
    env = make_navigation(num_agents=1)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config.with_ascii_uri(
        ascii_map, border_width=border_width
    )
    return make_nav_eval_env(env)


def ascii_eval_env(name: str, max_steps: int) -> EnvConfig:
    ascii_map = f"mettagrid/configs/maps/navigation/{name}.map"
    return make_ascii_env(max_steps=max_steps, ascii_map=ascii_map)


def make_nav_eval_env(env: EnvConfig) -> EnvConfig:
    env.game.agent.rewards.inventory.heart = 0.333
    return env


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


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
    return make_nav_eval_env(env)


def make_navigation_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(name="corridors", env=ascii_eval_env("corridors", 450)),
        SimulationConfig(
            name="cylinder_easy", env=ascii_eval_env("cylinder_easy", 250)
        ),
        SimulationConfig(name="cylinder", env=ascii_eval_env("cylinder", 250)),
        SimulationConfig(name="honeypot", env=ascii_eval_env("honeypot", 300)),
        SimulationConfig(name="knotty", env=ascii_eval_env("knotty", 500)),
        SimulationConfig(
            name="memory_palace", env=ascii_eval_env("memory_palace", 200)
        ),
        SimulationConfig(name="obstacles0", env=ascii_eval_env("obstacles0", 100)),
        SimulationConfig(name="obstacles1", env=ascii_eval_env("obstacles1", 300)),
        SimulationConfig(name="obstacles2", env=ascii_eval_env("obstacles2", 350)),
        SimulationConfig(name="obstacles3", env=ascii_eval_env("obstacles3", 300)),
        SimulationConfig(name="radial_large", env=ascii_eval_env("radial_large", 1000)),
        SimulationConfig(name="radial_mini", env=ascii_eval_env("radial_mini", 150)),
        SimulationConfig(name="radial_small", env=ascii_eval_env("radial_small", 120)),
        SimulationConfig(name="radial_maze", env=ascii_eval_env("radial_maze", 200)),
        SimulationConfig(name="swirls", env=ascii_eval_env("swirls", 350)),
        SimulationConfig(name="thecube", env=ascii_eval_env("thecube", 350)),
        SimulationConfig(name="walkaround", env=ascii_eval_env("walkaround", 250)),
        SimulationConfig(name="wanderout", env=ascii_eval_env("wanderout", 500)),
        SimulationConfig(
            name="emptyspace_outofsight", env=make_emptyspace_outofsight_env()
        ),
        SimulationConfig(name="walls_outofsight", env=make_walls_outofsight_env()),
        SimulationConfig(name="walls_sparse", env=make_walls_sparse_env()),
        SimulationConfig(name="walls_withinsight", env=make_walls_withinsight_env()),
        SimulationConfig(
            name="emptyspace_withinsight", env=make_emptyspace_withinsight_env()
        ),
        SimulationConfig(
            name="emptyspace_sparse_medium", env=make_emptyspace_sparse_medium_env()
        ),
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env()),
        SimulationConfig(name="labyrinth", env=make_labyrinth_env()),
    ]
