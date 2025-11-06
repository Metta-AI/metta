import copy
import functools
import importlib
import logging
import pkgutil
import typing

import fastapi
import pydantic.json_schema

import metta.cogworks.curriculum.curriculum
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid.base_config
import mettagrid.builder.envs
import mettagrid.map_builder.ascii
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scenes

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


def make_schemas_router() -> fastapi.APIRouter:
    router = fastapi.APIRouter(prefix="/schemas", tags=["schemas"])

    @router.get("/")
    @functools.cache
    def get_schemas() -> dict[str, typing.Any]:
        _, top_level_schema = pydantic.json_schema.models_json_schema(
            [
                (x, "serialization")
                for x in [
                    mettagrid.base_config.Config,  # including Config here guarantees that MapGen.Config name will be fully qualified
                    mettagrid.builder.envs.MettaGridConfig,
                    metta.sim.simulation_config.SimulationConfig,
                    metta.cogworks.curriculum.curriculum.CurriculumConfig,
                    metta.tools.play.PlayTool,
                    metta.tools.replay.ReplayTool,
                    metta.tools.eval.EvaluateTool,
                    metta.tools.train.TrainTool,
                    mettagrid.mapgen.mapgen.MapGen.Config,
                    mettagrid.builder.envs.RandomMapBuilder.Config,
                    mettagrid.map_builder.ascii.AsciiMapBuilder.Config,
                    *all_scenes(),
                ]
            ],
            title="Gridworks Schemas",
        )
        # I'm not sure if pydantic caches the schema, so copying just to be safe
        schemas = copy.deepcopy(top_level_schema)

        for name, definition in schemas["$defs"].items():
            if definition.get("title") == "Config":
                full_title = ".".join(name.split("__"))
                definition["title"] = full_title

        return schemas

    return router
