from typing import List

from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_navigation
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesParams
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityParams
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertParams
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected, MakeConnectedParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.types import AreaWhere


def make_mettagrid(width: int = 80, height: int = 80) -> MettaGridConfig:
    env = make_navigation(
        num_agents=4
    )  # reuse simple action config and objects (altar removed later if needed)
    env.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=Quadrants.factory(
            params=QuadrantsParams(base_size=11),
            children_actions=[
                # Top-left: City
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
                # Top-right: Forest
                ChildrenAction(
                    scene=BiomeForest.factory(
                        BiomeForestParams(clumpiness=4, seed_prob=0.05, growth_prob=0.6)
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.1"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Bottom-left: Caves
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
                # Bottom-right: Desert
                ChildrenAction(
                    scene=BiomeDesert.factory(
                        BiomeDesertParams(dune_period=8, ridge_width=2, angle=0.4)
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.3"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Global connectivity pass across quadrants
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="first",
                    lock="finalize",
                    limit=1,
                ),
            ],
        ),
    )
    return env


def make_evals() -> List[SimulationConfig]:
    env = make_mettagrid()
    return [SimulationConfig(suite="biomes", name="biomes_quadrants", env=env)]
