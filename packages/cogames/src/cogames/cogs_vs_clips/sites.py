"""Central site definitions shared across mission modules."""

from pydantic import Field

from cogames.cogs_vs_clips.mission_utils import get_map
from mettagrid.base_config import Config
from mettagrid.map_builder.map_builder import MapBuilderConfig


class Site(Config):
    name: str
    description: str
    map_builder: MapBuilderConfig

    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


# Evals site used by evaluation missions
EVALS = Site(
    name="evals",
    description="Evaluation missions for scripted agent testing",
    map_builder=get_map("evals/machina_eval_template.map"),
    min_cogs=1,
    max_cogs=5,
)
