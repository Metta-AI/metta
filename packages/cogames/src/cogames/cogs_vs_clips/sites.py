"""Central site definitions shared across mission modules."""

from cogames.cogs_vs_clips.mission import Site
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import MachinaArena, RandomTransform
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHub

# Fixed extractor hub sites (one per map size)
FIXED_30 = Site(
    name="fixed_30",
    description="Fixed extractor hub 30x30 map.",
    map_builder=get_map("evals/extractor_hub_30x30.map"),
    min_cogs=1,
    max_cogs=8,
)

FIXED_50 = Site(
    name="fixed_50",
    description="Fixed extractor hub 50x50 map.",
    map_builder=get_map("evals/extractor_hub_50x50.map"),
    min_cogs=1,
    max_cogs=8,
)

FIXED_70 = Site(
    name="fixed_70",
    description="Fixed extractor hub 70x70 map.",
    map_builder=get_map("evals/extractor_hub_70x70.map"),
    min_cogs=1,
    max_cogs=8,
)

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
    map_builder=MapGen.Config(width=200, height=200, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Evals site used by evaluation missions
# Note: Individual eval missions override this with their own specific maps
EVALS = Site(
    name="evals",
    description="Evaluation missions for scripted agent testing",
    map_builder=get_map("evals/eval_oxygen_bottleneck.map"),  # Default map (rarely used)
    min_cogs=1,
    max_cogs=8,
)

EASY_MODE = Site(
    name="easy_mode",
    description="Easy training configuration with simplified variants",
    map_builder=get_map("evals/extractor_hub_30x30.map"),
    min_cogs=1,
    max_cogs=8,
)

SITES = [
    FIXED_30,
    FIXED_50,
    FIXED_70,
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    EVALS,
    EASY_MODE,
]
