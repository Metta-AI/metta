"""Central site definitions shared across mission modules."""

from cogames.cogs_vs_clips.mission import Site
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import MachinaArena, RandomTransform
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHub

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

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    EVALS,
]
