import functools
import importlib
import logging
import pkgutil
from copy import deepcopy
from typing import Any

from fastapi import APIRouter
from pydantic.json_schema import models_json_schema

import mettagrid.mapgen.scenes
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.base_config import Config
from mettagrid.builder.envs import MettaGridConfig, RandomMapBuilder
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.mapgen import MapGen

logger = logging.getLogger(__name__)


def all_scenes():
    module_names = []
    for _, name, ispkg in pkgutil.walk_packages(
        mettagrid.mapgen.scenes.__path__,
        mettagrid.mapgen.scenes.__name__ + ".",
    ):
        if not ispkg:
            module_names.append(name)

    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if name.endswith("Config"):
                yield getattr(module, name)


def make_schemas_router() -> APIRouter:
    router = APIRouter(prefix="/schemas", tags=["schemas"])

    @router.get("/")
    @functools.cache
    def get_schemas() -> dict[str, Any]:
        _, top_level_schema = models_json_schema(
            [
                (x, "serialization")
                for x in [
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
                    *all_scenes(),
                ]
            ],
            title="Gridworks Schemas",
        )
        # I'm not sure if pydantic caches the schema, so copying just to be safe
        schemas = deepcopy(top_level_schema)

        for name, definition in schemas["$defs"].items():
            if definition.get("title") == "Config":
                full_title = ".".join(name.split("__"))
                definition["title"] = full_title

        return schemas

    return router
