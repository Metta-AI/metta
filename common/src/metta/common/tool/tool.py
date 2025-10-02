from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import ClassVar

from pydantic import Field, model_validator

from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config


class Tool(Config):
    """Base class for tools.

    To make a tool, you need to:
    1) Define a class that inherits from Tool.
    2) Define a `tool_name` class variable (e.g., "train", "evaluate").
    3) Define a `invoke` method that returns an exit code, optionally.
    4) Make a function that returns an instance of your tool class.
    5) Run the tool with `./tools/run.py <path.to.tool.function>`.

    The function can optionally take arguments.
    """

    # Required tool name for discovery and display; not a Pydantic field
    tool_name: ClassVar[str | None] = None

    system: SystemConfig = Field(default_factory=SystemConfig)

    @model_validator(mode="after")
    def validate_tool_name(self) -> "Tool":
        """Validate that concrete Tool classes define a tool_name."""
        # Only validate concrete tool classes (not intermediate base classes)
        if not inspect.isabstract(self.__class__) and self.__class__.tool_name is None:
            raise ValueError(f"Tool class {self.__class__.__name__} must define a 'tool_name' class variable")
        return self

    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
