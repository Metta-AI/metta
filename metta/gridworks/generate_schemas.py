#!/usr/bin/env -S uv run
import json
from copy import deepcopy

from pydantic.json_schema import models_json_schema

from metta.sim.simulation_config import SimulationConfig
from mettagrid.base_config import Config
from mettagrid.builder.envs import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen


def main():
    _, top_level_schema = models_json_schema(
        [
            (x, "serialization")
            for x in [
                Config,  # including Config here guarantees that MapGen.Config name will be fully qualified
                MettaGridConfig,
                SimulationConfig,
                MapGen.Config,
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
