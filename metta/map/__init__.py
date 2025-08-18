# metta/map/__init__.py

"""Map builder module for MettaGrid."""

from typing import Annotated, Union

from pydantic import Field

from metta.map.mapgen import MapGenConfig
from metta.map.terrain_from_numpy import TerrainFromNumpyConfig
from metta.mettagrid.map_builder.ascii import AsciiMapBuilderConfig

# Define the discriminated union here to avoid circular imports
MapGenConfigUnion = Annotated[
    Union[
        Annotated[AsciiMapBuilderConfig, Field(discriminator="type")],
        Annotated[MapGenConfig, Field(discriminator="type")],
        Annotated[TerrainFromNumpyConfig, Field(discriminator="type")],
    ],
    Field(discriminator="type"),
]

# Rebuild the model now that MapGenConfigUnion is defined
MapGenConfig.model_rebuild()

__all__ = ["MapGenConfigUnion"]
