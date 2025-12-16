import functools
import importlib
import logging
import pkgutil
from copy import deepcopy
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

import mettagrid.mapgen.scenes
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.base_config import Config
from mettagrid.builder.envs import MettaGridConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

logger = logging.getLogger(__name__)


def all_scenes():
    for _, name, ispkg in pkgutil.walk_packages(
        mettagrid.mapgen.scenes.__path__,
        mettagrid.mapgen.scenes.__name__ + ".",
    ):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Failed to import scene module {name}: {e}")
            continue

        for obj_name in dir(module):
            if not obj_name.endswith("Config"):
                continue
            obj = getattr(module, obj_name)
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                yield obj
            else:
                logger.debug(f"Skipping non-Pydantic model from scene: {name}.{obj_name} = {type(obj)}")


def get_schemas() -> dict[str, Any]:
    models = [
        Config,  # including Config here guarantees that MapGen.Config name will be fully qualified
        MettaGridConfig,
        SimulationConfig,
        CurriculumConfig,
        PlayTool,
        ReplayTool,
        EvaluateTool,
        TrainTool,
        MapGen.Config,
        RandomMapBuilder.Config,
        AsciiMapBuilder.Config,
        Mission,
        Site,
        MissionVariant,
        *all_scenes(),
    ]

    # Sanity check: ensure all models are pydantic models
    valid_models = []
    for m in models:
        if isinstance(m, type) and issubclass(m, BaseModel):
            valid_models.append(m)
        else:
            logger.warning(f"Skipping non-Pydantic model from schema: {m}")

    _, top_level_schema = models_json_schema(
        [(m, "serialization") for m in valid_models],
        title="Gridworks Schemas",
    )

    # I'm not sure if pydantic caches the schema, so copying just to be safe
    schemas = deepcopy(top_level_schema)

    for name, definition in schemas["$defs"].items():
        if definition.get("title") == "Config":
            full_title = ".".join(name.split("__"))
            definition["title"] = full_title

    return schemas


def make_schemas_router() -> APIRouter:
    router = APIRouter(prefix="/schemas", tags=["schemas"])

    @router.get("/")
    @functools.cache
    def get_schemas_route() -> dict[str, Any]:
        return get_schemas()

    return router
