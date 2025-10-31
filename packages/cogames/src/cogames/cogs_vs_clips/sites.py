"""Central site definitions shared across mission modules."""

from pydantic import Field

from cogames.cogs_vs_clips.procedural import MachinaArenaConfig
from mettagrid.base_config import Config
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen


class Site(Config):
    name: str
    description: str
    map_builder: MapBuilderConfig

    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


# Evals site used by evaluation missions
# Note: Individual eval missions override this with their own maps
EVALS = Site(
    name="evals",
    description="Evaluation missions for scripted agent testing",
    map_builder=MapGen.Config(width=50, height=50, instance=MachinaArenaConfig(spawn_count=8)),
    min_cogs=1,
    max_cogs=8,
)
