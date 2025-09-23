from abc import abstractmethod
from typing import ClassVar, List

from pydantic import Field

from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config


class Tool(Config):
    """Base class for tools.

    To make a tool, you need to:
    1) Define a class that inherits from Tool.
    2) Define a `invoke` method that returns an exit code, optionally.
    3) Make a function that returns an instance of your tool class.
    4) Run the tool with `./tools/run.py <path.to.tool.function>`.

    The function can optionally take arguments."""

    # Optional canonical name for discovery and display; not a Pydantic field
    tool_name: ClassVar[str | None] = None
    tool_aliases: ClassVar[List[str]] = []

    system: SystemConfig = Field(default_factory=SystemConfig)

    # Returns exit code, optionally.
    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
