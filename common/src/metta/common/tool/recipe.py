"""Recipe abstraction for tool discovery and inference.

A Recipe represents a module that defines tools either explicitly (via functions)
or implicitly (via mettagrid()/simulations() that can be inferred into tools).
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from typing_extensions import get_type_hints

from metta.common.tool import Tool
from metta.common.tool.tool_registry import get_tool_registry
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig


class Recipe:
    """Represents a recipe module that can provide tools."""

    def __init__(self, module: ModuleType):
        self.module = module
        self.module_name = module.__name__
        # Build tool maps on initialization
        self._name_to_tool: dict[str, Callable[[], Tool]] = {}
        self._canonical_to_tools: dict[str, list[tuple[str, Callable[[], Tool]]]] = {}
        self._build_tool_maps()

    @property
    def short_name(self) -> str:
        """Get short name without 'experiments.recipes.' prefix."""
        return self.module_name.replace("experiments.recipes.", "")

    def _build_tool_maps(self) -> None:
        """Build name->tool and canonical->tools maps for efficient lookup."""

        registry = get_tool_registry()

        # Get explicit tools
        explicit_tools = self.get_explicit_tools()

        # Build both maps
        for func_name, func in explicit_tools.items():
            # Add to name->tool map
            self._name_to_tool[func_name] = func

            # Determine which tool class this function returns
            try:
                hints = get_type_hints(func)
                return_type = hints.get("return")
                if return_type and isinstance(return_type, type) and issubclass(return_type, Tool):
                    # Get canonical name for this tool class
                    tool_name = return_type.tool_name
                    if tool_name not in self._canonical_to_tools:
                        self._canonical_to_tools[tool_name] = []
                    self._canonical_to_tools[tool_name].append((func_name, func))
            except Exception:
                pass

        # Add inferred tools to both maps
        for canonical_name, tool_class in registry.get_all_tools().items():
            inferred = self.infer_tool(tool_class)
            if inferred is not None and canonical_name not in self._canonical_to_tools:
                # Add inferred tool (no specific function name, use canonical)
                self._name_to_tool[canonical_name] = inferred
                self._canonical_to_tools[canonical_name] = [(canonical_name, inferred)]

    @classmethod
    def load(cls, module_path: str) -> Optional["Recipe"]:
        """Try to load a recipe from a module path. e.g. 'experiments.recipes.arena'"""
        try:
            module = importlib.import_module(module_path)
            return cls(module)
        except ImportError:
            return None

    def get_explicit_tools(self) -> dict[str, Callable[[], Tool]]:
        """Returns only tools explicitly defined in this recipe."""

        tools: dict[str, Callable[[], Tool]] = {}

        for name in dir(self.module):
            if name.startswith("_"):
                continue

            attr = getattr(self.module, name)
            if not callable(attr) or isinstance(attr, type):
                continue

            try:
                hints = get_type_hints(attr)
                return_type = hints.get("return")
                if return_type and isinstance(return_type, type) and issubclass(return_type, Tool):
                    tools[name] = attr  # type: ignore
            except Exception:
                pass

        return tools

    def get_configs(self) -> tuple[MettaGridConfig | None, list[SimulationConfig] | None]:
        """Get mettagrid() and simulations() from this recipe, going through a global cache."""
        return _get_recipe_configs(self.module)

    def infer_tool(self, tool_class: type[Tool]) -> Optional[Callable[[], Tool]]:
        """Infer a tool instance from this recipe's configs.

        Args:
            tool_class: Tool class to infer

        Returns:
            Tool maker function or None
        """
        # Skip if tool doesn't support inference
        if tool_class.infer == Tool.infer:
            return None

        mg, sims = self.get_configs()
        if mg is None and sims is None:
            return None

        tool = tool_class.infer(mettagrid=mg, simulations=sims)
        if tool is None:
            return None

        return lambda: tool

    def get_all_tool_names(self) -> set[str]:
        """Get all tool names available from this recipe (explicit + inferred)."""
        # Use the pre-built maps
        return set(self._name_to_tool.keys())

    def supports_tool(self, canonical_tool_name: str) -> bool:
        """Check if this recipe supports a tool with the given canonical name."""
        return canonical_tool_name in self._canonical_to_tools

    def get_tools_for_canonical(self, canonical_tool_name: str) -> list[tuple[str, Callable[[], Tool]]]:
        """Get all tools (by name and callable) that return the given canonical tool type."""
        return self._canonical_to_tools.get(canonical_tool_name, [])


@lru_cache(maxsize=128)
def _get_recipe_configs(module: ModuleType) -> tuple[MettaGridConfig | None, list[SimulationConfig] | None]:
    """Get mettagrid() and simulations() from a recipe module.

    Cached to avoid repeated construction.
    Returns (mettagrid, simulations) tuple.
    """
    # Try mettagrid()
    mg = None
    if hasattr(module, "mettagrid") and callable(module.mettagrid):
        try:
            result = module.mettagrid()
            if isinstance(result, MettaGridConfig):
                mg = result
        except Exception:
            pass

    # Try simulations()
    sims = None
    if hasattr(module, "simulations") and callable(module.simulations):
        try:
            result = module.simulations()
            if isinstance(result, (list, tuple)) and result:
                if isinstance(result[0], SimulationConfig):
                    sims = list(result)
        except Exception:
            pass

    return mg, sims
