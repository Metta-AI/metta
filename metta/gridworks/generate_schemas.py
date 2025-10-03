#!/usr/bin/env -S uv run
import importlib
import json
import pkgutil
from copy import deepcopy

from pydantic.json_schema import models_json_schema

import mettagrid.mapgen.scenes
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.base_config import Config
from mettagrid.builder.envs import MettaGridConfig, RandomMapBuilder
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.mapgen import MapGen


def all_scenes():
    module_names = []
    for _, name, ispkg in pkgutil.walk_packages(
        mettagrid.mapgen.scenes.__path__,
        mettagrid.mapgen.scenes.__name__ + ".",
    ):
        if not ispkg:
            module_names.append(name)

    # (Optional) Import them

    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if name.endswith("Config"):
                yield getattr(module, name)


def main():
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
                SimTool,
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
    top_level_schema = deepcopy(top_level_schema)

    for name, definition in top_level_schema["$defs"].items():
        if definition.get("title") == "Config":
            full_title = ".".join(name.split("__"))
            definition["title"] = full_title

    print(json.dumps(top_level_schema, indent=2))


if __name__ == "__main__":
    main()
