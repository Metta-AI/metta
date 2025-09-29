from typing import List

from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_navigation
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubParams
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesParams
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityParams
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertParams
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestParams
from mettagrid.mapgen.scenes.bsp import BSP, BSPLayout, BSPLayoutParams, BSPParams
from mettagrid.mapgen.scenes.fill_area import FillArea, FillAreaParams
from mettagrid.mapgen.scenes.layout import Layout, LayoutArea, LayoutParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected, MakeConnectedParams
from mettagrid.mapgen.scenes.maze import Maze, MazeParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.scenes.radial_maze import RadialMaze, RadialMazeParams
from mettagrid.mapgen.types import AreaWhere


def make_mettagrid(
    width: int = 500, height: int = 500
) -> tuple[
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
]:
    desert_noise = make_navigation(num_agents=4)
    desert_noise.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeDesert.factory(
            BiomeDesertParams(dune_period=8, ridge_width=2, angle=0.4, noise_prob=0.6)
        ),
    )

    city = make_navigation(num_agents=4)
    city.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCity.factory(
            BiomeCityParams(pitch=10, road_width=2, jitter=2, place_prob=0.9)
        ),
    )

    forest = make_navigation(num_agents=4)
    forest.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeForest.factory(
            BiomeForestParams(clumpiness=4, seed_prob=0.05, growth_prob=0.6)
        ),
    )

    caves = make_navigation(num_agents=4)
    caves.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(
            BiomeCavesParams(fill_prob=0.45, steps=4, birth_limit=5, death_limit=3)
        ),
    )

    astroid = make_navigation(num_agents=4)
    astroid.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(
            BiomeCavesParams(
                fill_prob=0.4,
                steps=5,
                birth_limit=5,
                death_limit=3,
            ),
            children_actions=[
                # Sanctum outpost: orderly ring around central hub
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=120,
                                    height=120,
                                    placement="center",
                                    tag="sanctum.outpost",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=FillArea.factory(FillAreaParams(value="empty")),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCity.factory(
                                    BiomeCityParams(
                                        pitch=12,
                                        road_width=3,
                                        jitter=1,
                                        place_prob=0.6,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCaves.factory(
                                    BiomeCavesParams(
                                        fill_prob=0.25,
                                        steps=2,
                                        birth_limit=4,
                                        death_limit=3,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    lock="sanctum.outpost",
                    limit=1,
                ),
                # Stencil a constellation of pockets across the caves using BSP zones.
                ChildrenAction(
                    scene=BSPLayout.factory(
                        BSPLayoutParams(area_count=12),
                        children_actions=[
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=90,
                                                height=90,
                                                placement="center",
                                                tag="astroid.desert",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=8,
                                                    ridge_width=2,
                                                    angle=0.4,
                                                    noise_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=30,
                                                            height=30,
                                                            placement="center",
                                                            tag="astroid.radial",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=RadialMaze.factory(
                                                            RadialMazeParams(
                                                                arms=10,
                                                                arm_width=2,
                                                                arm_length=28,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone.bsp",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=60,
                                                placement="center",
                                                tag="astroid.forest",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=4,
                                                    seed_prob=0.05,
                                                    growth_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=24,
                                                            height=24,
                                                            placement="center",
                                                            tag="astroid.bsp",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BSP.factory(
                                                            BSPParams(
                                                                rooms=8,
                                                                min_room_size=3,
                                                                min_room_size_ratio=0.4,
                                                                max_room_size_ratio=0.8,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone.radial",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=60,
                                                placement="center",
                                                tag="astroid.city.maze1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=30,
                                                            height=30,
                                                            placement="center",
                                                            tag="astroid.maze",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="dfs",
                                                                room_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=1
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone.city1",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=36,
                                                height=36,
                                                placement="center",
                                                tag="astroid.city.maze2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=28,
                                                            height=28,
                                                            placement="center",
                                                            tag="astroid.maze.kruskal",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="kruskal",
                                                                room_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=1
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone.city2",
                            ),
                        ],
                    ),
                    where="full",
                    limit=1,
                ),
                # Central sanctum retained as the cave nexus, stamped last.
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="altar",
                                        include_inner_wall=True,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
            ],
        ),
    )

    env = make_navigation(num_agents=4)
    # reuse simple action config and objects (altar removed later if needed)
    env.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=Quadrants.factory(
            params=QuadrantsParams(base_size=6),  # Creates 4 quadrants for biomes (0-3)
            # Dungeons use BSP layout to create 3 bounded zones
            children_actions=[
                # Top-left: City biome
                ChildrenAction(
                    scene=BiomeCity.factory(
                        BiomeCityParams(
                            pitch=10, road_width=2, jitter=2, place_prob=0.9
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.0"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Top-right: Forest biome
                ChildrenAction(
                    scene=BiomeForest.factory(
                        BiomeForestParams(clumpiness=4, seed_prob=0.05, growth_prob=0.6)
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.1"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Bottom-left: Caves biome
                ChildrenAction(
                    scene=BiomeCaves.factory(
                        BiomeCavesParams(
                            fill_prob=0.45, steps=4, birth_limit=5, death_limit=3
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.2"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Fractal-style dungeon embedded in caves quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=50,
                                    height=50,
                                    placement="center",
                                    tag="dz_fractal",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="kruskal",
                                        room_size=IntConstantDistribution(value=2),
                                        wall_size=IntConstantDistribution(value=1),
                                    )
                                ),
                                where=AreaWhere(tags=["dz_fractal"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.2"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Bottom-right: Desert biome
                ChildrenAction(
                    scene=BiomeDesert.factory(
                        BiomeDesertParams(
                            dune_period=8, ridge_width=2, angle=0.4, noise_prob=0.6
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.3"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # BSP Dungeon - centered in city quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=70,
                                    height=60,
                                    placement="center",
                                    tag="dz_bsp",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=8,
                                        min_room_size=3,
                                        min_room_size_ratio=0.4,
                                        max_room_size_ratio=0.8,
                                    )
                                ),
                                where=AreaWhere(tags=["dz_bsp"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.0"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Maze Dungeon - centered in forest quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=50,
                                    height=50,
                                    placement="center",
                                    tag="dz_maze",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="dfs",
                                        room_size=IntConstantDistribution(value=3),
                                        wall_size=IntConstantDistribution(value=1),
                                    )
                                ),
                                where=AreaWhere(tags=["dz_maze"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.1"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Radial Maze Dungeon - centered in desert quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=80,
                                    height=80,
                                    placement="center",
                                    tag="dz_radial",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=10, arm_width=2, arm_length=40
                                    )
                                ),
                                where=AreaWhere(tags=["dz_radial"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.3"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Global connectivity pass - runs after everything to connect dungeons and biomes
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
                # Central sanctum stamp: carve centered area then place BaseHub inside it
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="altar",
                                        include_inner_wall=True,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
            ],
        ),
    )

    # BSP Dungeon environment - standalone roguelike dungeon (15x15 to 25x25)
    bsp_dungeon = make_navigation(num_agents=4)
    bsp_dungeon.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BSP.factory(
            BSPParams(
                rooms=10,
                min_room_size=4,
                min_room_size_ratio=0.4,
                max_room_size_ratio=0.9,
            )
        ),
    )

    # Radial Maze Dungeon environment - standalone fractal maze (12x12 to 20x20)
    radial_maze = make_navigation(num_agents=4)
    radial_maze.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=RadialMaze.factory(RadialMazeParams(arms=8, arm_width=3, arm_length=15)),
    )

    return env, desert_noise, city, caves, forest, bsp_dungeon, radial_maze, astroid


def make_evals() -> List[SimulationConfig]:
    env, desert_noise, city, caves, forest, bsp_dungeon, radial_maze, astroid = (
        make_mettagrid()
    )
    return [
        SimulationConfig(suite="biomes", name="biomes_quadrants", env=env),
        SimulationConfig(suite="biomes", name="desert_noise", env=desert_noise),
        SimulationConfig(suite="biomes", name="city", env=city),
        SimulationConfig(suite="biomes", name="caves", env=caves),
        SimulationConfig(suite="biomes", name="forest", env=forest),
        SimulationConfig(suite="biomes", name="bsp_dungeon", env=bsp_dungeon),
        SimulationConfig(suite="biomes", name="radial_maze", env=radial_maze),
        SimulationConfig(suite="biomes", name="astroid", env=astroid),
    ]
