from typing import Any

from pydantic import Field

from mettagrid.config import Config


class ObsConfig(Config):
    width: int = Field(default=11)
    height: int = Field(default=11)
    token_dim: int = Field(default=3)
    num_tokens: int = Field(default=200)
    features: list[Any] = Field(default_factory=list)
