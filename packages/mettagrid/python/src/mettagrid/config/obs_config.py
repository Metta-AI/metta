"""Observation configuration.

Feature IDs and names are managed by IdMap.
Changing feature IDs will break models trained on old feature IDs.
"""

import pydantic

import mettagrid.base_config


class ObsConfig(mettagrid.base_config.Config):
    """Observation configuration."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    width: int = pydantic.Field(default=11)
    height: int = pydantic.Field(default=11)
    token_dim: int = pydantic.Field(default=3)
    num_tokens: int = pydantic.Field(default=200)
