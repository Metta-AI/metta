"""Recipe abstraction for tool discovery.

A Recipe represents a module that defines tool makers - functions that return tool instances.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any, Callable, Optional

from typing_extensions import TypeIs, get_type_hints

from metta.common.tool import Tool

ToolMaker = Callable[..., Tool]


def is_tool_maker(obj: Any) -> TypeIs[ToolMaker]:
    """Type guard to check if an object is a tool maker function.

    A tool maker is a callable that returns a Tool instance.
    """
    if not callable(obj) or isinstance(obj, type):
        return False

    try:
        hints = get_type_hints(obj)
        return_type = hints.get("return")
        return return_type is not None and isinstance(return_type, type) and issubclass(return_type, Tool)
    except Exception:
        return False


class Recipe:
    """Represents a recipe module that can provide tool makers."""

    def __init__(self, module: ModuleType):
        self.module = module
        self.module_name = module.__name__
        # Build tool maker map on initialization: maker_name -> tool_maker
        self._maker_name_to_tool_maker: dict[str, ToolMaker] = {}
        # Also build reverse map: tool_type -> list of (maker_name, tool_maker)
        self._tool_type_to_makers: dict[str, list[tuple[str, ToolMaker]]] = {}
        self._build_tool_maps()

    @property
    def short_name(self) -> str:
        return self.module_name.replace("experiments.recipes.", "")

    def _build_tool_maps(self) -> None:
        """Build maker_name->tool_maker and tool_class_name->makers maps."""
        # Get explicit tool makers
        explicit_makers = self.get_explicit_tool_makers()

        # Build both maps from explicit tool makers
        for maker_name, maker_func in explicit_makers.items():
            # Add to maker_name->tool_maker map
            self._maker_name_to_tool_maker[maker_name] = maker_func

            # Determine which tool type this maker returns
            try:
                hints = get_type_hints(maker_func)
                return_type = hints.get("return")
                if return_type and isinstance(return_type, type) and issubclass(return_type, Tool):
                    tool_type = return_type.tool_type_name()
                    if tool_type not in self._tool_type_to_makers:
                        self._tool_type_to_makers[tool_type] = []
                    self._tool_type_to_makers[tool_type].append((maker_name, maker_func))
            except Exception:
                pass

    @classmethod
    def load(cls, module_path: str) -> Optional["Recipe"]:
        """Try to load a recipe from a module path. e.g. 'experiments.recipes.arena'"""
        if importlib.util.find_spec(module_path) is not None:
            module = importlib.import_module(module_path)
            return cls(module)
        return None

    def get_explicit_tool_makers(self) -> dict[str, ToolMaker]:
        """Returns only tool makers explicitly defined in this recipe."""
        makers: dict[str, ToolMaker] = {}

        for name in dir(self.module):
            if name.startswith("_"):
                continue

            attr = getattr(self.module, name)
            if is_tool_maker(attr):
                makers[name] = attr

        return makers

    def get_all_tool_maker_names(self) -> set[str]:
        """Get all tool maker names available from this recipe."""
        return set(self._maker_name_to_tool_maker.keys())

    def get_tool_maker(self, name: str) -> ToolMaker | None:
        """Get a tool maker by maker name or tool type.

        Args:
            name: Either a maker name (e.g., 'replay_null', 'train_shaped')
                  or a tool type (e.g., 'evaluate', 'train')

        Returns:
            Tool maker, or None if not found
        """
        # Try direct maker name lookup first
        if name in self._maker_name_to_tool_maker:
            return self._maker_name_to_tool_maker[name]

        # Try tool type lookup (returns first matching maker)
        makers = self._tool_type_to_makers.get(name, [])
        if makers:
            return makers[0][1]

        return None

    def get_makers_for_tool(self, tool_type: str) -> list[tuple[str, ToolMaker]]:
        """Get all tool makers that return the given tool type.

        Useful for listing all implementations of a tool type (e.g., 'train', 'train_shaped').

        Args:
            tool_type: Tool type identifier (e.g., 'train', 'evaluate')

        Returns:
            List of (maker_name, tool_maker) tuples
        """
        return self._tool_type_to_makers.get(tool_type, [])
