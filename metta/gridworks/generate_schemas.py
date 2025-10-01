#!/usr/bin/env -S uv run
import json

from pydantic.json_schema import models_json_schema

from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder.envs import MettaGridConfig
from mettagrid.config import Config
from mettagrid.mapgen.mapgen import MapGen

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
print(json.dumps(top_level_schema, indent=2))
