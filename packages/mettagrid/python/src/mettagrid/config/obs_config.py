"""Observation configuration.

Feature IDs and names are managed by IdMap.
Changing feature IDs will break models trained on old feature IDs.
"""

from pydantic import ConfigDict, Field

from mettagrid.base_config import Config


class ObsConfig(Config):
    """Observation configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    width: int = Field(default=11)
    height: int = Field(default=11)
    token_dim: int = Field(default=3)
    num_tokens: int = Field(default=200)
    token_value_max: int = Field(default=255)
    """Maximum value per inventory token (base for encoding). Default 255 for efficient byte packing."""
