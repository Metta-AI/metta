"""Recipe abstraction for tool discovery and inference.

A Recipe represents a module that defines tools either explicitly (via functions)
or implicitly (via mettagrid()/simulations() that can be inferred into tools).
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from typing_extensions import get_type_hints

from metta.common.tool import Tool
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig


class Recipe:
    """Represents a recipe module that can provide tools."""

    def __init__(self, module: ModuleType):
        self.module = module
        self.module_name = module.__name__

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
        from metta.common.tool.tool_registry import get_tool_registry

        tool_names = set(self.get_explicit_tools().keys())

        # Add inferred tools
        registry = get_tool_registry()
        for canonical_name, tool_class in registry.get_all_tools().items():
            if canonical_name not in tool_names and self.infer_tool(tool_class) is not None:
                tool_names.add(canonical_name)

        return tool_names

    @staticmethod
    def discover_all(base_package: str = "experiments.recipes") -> list[Recipe]:
        """Discover all recipe modules under a base package.

        Args:
            base_package: Base package to search for recipes (default: experiments.recipes)

        Returns:
            List of Recipe instances
        """
        recipes = []

        try:
            base_module = importlib.import_module(base_package)
        except ImportError:
            return recipes

        # Get the package path
        if hasattr(base_module, "__path__"):
            package_paths = base_module.__path__
        else:
            return recipes

        # Walk through all modules in the package
        for _, modname, _ in pkgutil.walk_packages(package_paths, prefix=f"{base_package}."):
            # Skip private modules
            if modname.split(".")[-1].startswith("_"):
                continue

            recipe = Recipe.load(modname)
            if recipe:
                # Only include if it has tools or configs
                if recipe.get_explicit_tools() or recipe.get_configs() != (None, None):
                    recipes.append(recipe)

        return recipes


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
