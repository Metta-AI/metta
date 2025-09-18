from abc import abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from metta.rl.system_config import SystemConfig
from mettagrid.config import Config

TConfig = TypeVar("TConfig", bound=BaseModel)


class Tool(Config, Generic[TConfig]):
    """Base tool class with typed config field.

    To make a tool, you need to:
    1) Define a class that inherits from Tool[ConfigType] where ConfigType is your config type.
    2) Define an `invoke` method that returns an exit code, optionally.
    3) Use the recipe running system: tools are invoked via `./tools/run.py <verb> <recipe>`.

    The function can optionally take arguments."""

    system: SystemConfig = Field(default_factory=SystemConfig)
    config: TConfig  # The type annotation is the entire contract

    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
