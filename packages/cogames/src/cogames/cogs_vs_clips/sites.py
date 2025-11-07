"""Central site definitions shared across mission modules."""

import cogames.cogs_vs_clips.mission
import cogames.cogs_vs_clips.mission_utils
import cogames.cogs_vs_clips.procedural
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scenes.base_hub

TRAINING_FACILITY = cogames.cogs_vs_clips.mission.Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=mettagrid.mapgen.mapgen.MapGen.Config(
        width=13,
        height=13,
        instance=cogames.cogs_vs_clips.procedural.RandomTransform.Config(
            scene=mettagrid.mapgen.scenes.base_hub.BaseHub.Config(
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

HELLO_WORLD = cogames.cogs_vs_clips.mission.Site(
    name="hello_world",
    description="Welcome to space.",
    map_builder=mettagrid.mapgen.mapgen.MapGen.Config(
        width=100, height=100, instance=cogames.cogs_vs_clips.procedural.MachinaArena.Config(spawn_count=20)
    ),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = cogames.cogs_vs_clips.mission.Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=mettagrid.mapgen.mapgen.MapGen.Config(
        width=200, height=200, instance=cogames.cogs_vs_clips.procedural.MachinaArena.Config(spawn_count=20)
    ),
    min_cogs=1,
    max_cogs=20,
)

# Evals site used by evaluation missions
# Note: Individual eval missions override this with their own specific maps
EVALS = cogames.cogs_vs_clips.mission.Site(
    name="evals",
    description="Evaluation missions for scripted agent testing",
    map_builder=cogames.cogs_vs_clips.mission_utils.get_map(
        "evals/eval_oxygen_bottleneck.map"
    ),  # Default map (rarely used)
    min_cogs=1,
    max_cogs=8,
)

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    EVALS,
]
