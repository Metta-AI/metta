"""Recipe abstraction for tool discovery.

A Recipe represents a module that defines tools via functions that return Tool instances.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable, Optional

from typing_extensions import get_type_hints

from metta.common.tool import Tool


class Recipe:
    """Represents a recipe module that can provide tools."""

    def __init__(self, module: ModuleType):
        self.module = module
        self.module_name = module.__name__
        # Build tool map on initialization: function_name -> tool_maker
        self._name_to_tool: dict[str, Callable[[], Tool]] = {}
        # Also build reverse map: tool_name -> list of (function_name, tool_maker)
        self._tool_name_to_functions: dict[str, list[tuple[str, Callable[[], Tool]]]] = {}
        self._build_tool_maps()

    @property
    def short_name(self) -> str:
        return self.module_name.replace("experiments.recipes.", "")

    def _build_tool_maps(self) -> None:
        """Build function_name->tool and tool_name->functions maps."""
        # Get explicit tools
        explicit_tools = self.get_explicit_tools()

        # Build both maps from explicit tools
        for func_name, func in explicit_tools.items():
            # Add to name->tool map
            self._name_to_tool[func_name] = func

            # Determine which tool class this function returns
            try:
                hints = get_type_hints(func)
                return_type = hints.get("return")
                if return_type and isinstance(return_type, type) and issubclass(return_type, Tool):
                    tool_name = return_type.tool_name
                    if tool_name not in self._tool_name_to_functions:
                        self._tool_name_to_functions[tool_name] = []
                    self._tool_name_to_functions[tool_name].append((func_name, func))
            except Exception:
                pass

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

    def get_all_tool_names(self) -> set[str]:
        """Get all tool function names available from this recipe."""
        return set(self._name_to_tool.keys())

    def get_tool(self, name: str) -> Callable[[], Tool] | None:
        """Get a tool by function name or tool name.

        Args:
            name: Either a function name (e.g., 'replay_null', 'train_shaped')
                  or a tool name (e.g., 'evaluate', 'train')

        Returns:
            Tool maker function, or None if not found
        """
        # Try direct function name lookup first
        if name in self._name_to_tool:
            return self._name_to_tool[name]

        # Try tool name lookup (returns first matching function)
        functions = self._tool_name_to_functions.get(name, [])
        if functions:
            return functions[0][1]

        return None

    def get_functions_for_tool(self, tool_name: str) -> list[tuple[str, Callable[[], Tool]]]:
        """Get all functions that return the given tool type.

        Useful for listing all implementations of a tool (e.g., 'train', 'train_shaped').
        """
        return self._tool_name_to_functions.get(tool_name, [])
