from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import ClassVar, List

from pydantic import Field

from metta.rl.system_config import SystemConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.base_config import Config


class Tool(Config):
    """Base class for tools.

    To make a tool, you need to:
    1) Define a class that inherits from Tool.
    2) Define a `tool_name` class variable with the canonical name.
    3) Define a `invoke` method that returns an exit code, optionally.
    4) Make a function that returns an instance of your tool class.
    5) Run the tool with `./tools/run.py <path.to.tool.function>`.

    The function can optionally take arguments.

    Inferred tool support:
    Override the `infer` class method to enable automatic tool generation
    from recipe modules that define `mettagrid()` or `simulations()` functions.
    If not overridden, the tool cannot be inferred from recipes.
    """

    # Required canonical name for discovery and display; not a Pydantic field
    tool_name: ClassVar[str | None] = None
    tool_aliases: ClassVar[List[str]] = []

    system: SystemConfig = Field(default_factory=SystemConfig)

    def __init_subclass__(cls, **kwargs):
        """Validate that all Tool subclasses define a tool_name."""
        super().__init_subclass__(**kwargs)
        # Only validate concrete tool classes (not intermediate base classes)
        if not inspect.isabstract(cls) and cls.tool_name is None:
            raise TypeError(f"Tool class {cls.__name__} must define a 'tool_name' class variable")

    @classmethod
    def infer(
        cls,
        mettagrid: MettaGridConfig | None = None,
        simulations: list[SimulationConfig] | None = None,
    ) -> Tool | None:
        """Generate a tool instance from recipe configurations.

        Override this method to support automatic inference from recipes.

        Args:
            mettagrid: The base environment configuration from recipe.mettagrid()
            simulations: Simulation configs from recipe.simulations()

        Returns:
            A configured tool instance, or None if inference is not supported
        """
        return None  # By default, tools don't support inference

    # Returns exit code, optionally.
    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
