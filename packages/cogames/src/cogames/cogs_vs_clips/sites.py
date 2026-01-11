"""Central site definitions shared across mission modules."""

from cogames.cogs_vs_clips.mission import Site
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import MachinaArena, RandomTransform, SequentialMachinaArena
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubConfig

TRAINING_FACILITY = Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=MapGen.Config(
        width=13,
        height=13,
        instance=RandomTransform.Config(
            scene=BaseHub.Config(
                spawn_count=4,
                corner_bundle="extractors",
                corner_objects=[
                    "carbon_extractor",
                    "oxygen_extractor",
                    "germanium_extractor",
                    "silicon_extractor",
                ],
                cross_bundle="none",
            )
        ),
    ),
    min_cogs=1,
    max_cogs=4,
)

HELLO_WORLD = Site(
    name="hello_world",
    description="Welcome to space.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=MapGen.Config(width=88, height=88, instance=SequentialMachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Evals site used by diagnostic evaluation missions
# Note: Individual diagnostic missions override this with their own specific maps
EVALS = Site(
    name="evals",
    description="Diagnostic evaluation arenas.",
    map_builder=get_map("diagnostic_evals/diagnostic_radial.map"),  # Default map (rarely used)
    min_cogs=1,
    max_cogs=8,
)


def make_cogsguard_arena_site(num_agents: int = 10) -> Site:
    """Create a CogsGuard arena site with configurable agent count."""
    hub_config = BaseHubConfig(
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
        stations=[
            "aligner_station",
            "scrambler_station",
            "miner_station",
            "scout_station",
            "chest",
        ],
    )
    map_builder = MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
            hub=hub_config,
        ),
    )
    return Site(
        name="cogsguard_arena",
        description="CogsGuard arena map",
        map_builder=map_builder,
        min_cogs=num_agents,
        max_cogs=num_agents,
    )


# Default CogsGuard arena site
COGSGUARD_ARENA = make_cogsguard_arena_site(num_agents=10)


SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    EVALS,
    COGSGUARD_ARENA,
]
