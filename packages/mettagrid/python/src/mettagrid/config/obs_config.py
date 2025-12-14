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
    token_value_base: int = Field(default=256)
    """Base for multi-token inventory encoding (value per token: 0 to base-1).

    Default 256 for efficient byte packing.
    """
