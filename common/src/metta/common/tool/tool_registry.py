"""Tool registry for managing tool types and aliases.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class manages tool types (classes) and their aliases.
Tool loading and path resolution is in tool.py.
"""

from __future__ import annotations

from metta.common.tool import Tool
from metta.tools.analyze import AnalysisTool
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Tools are explicitly registered on module import (see registration section below)."""

    _instance: ToolRegistry | None = None
    _registry: dict[str, type[Tool]]
    _tool_aliases_cache: dict[str, list[str]] | None = None
    _tool_name_map_cache: dict[str, str] | None = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._tool_aliases_cache = None
            cls._instance._tool_name_map_cache = None
        return cls._instance

    def register(self, tool_class: type[Tool]) -> None:
        tool_name = tool_class.tool_name
        assert tool_name is not None, f"Tool class {tool_class.__name__} required by tool validation"
        self._registry[tool_name] = tool_class
        # Invalidate caches when registry changes
        self._tool_aliases_cache = None
        self._tool_name_map_cache = None

    def get_all_tools(self) -> dict[str, type[Tool]]:
        """Get all registered tool classes mapped by their canonical name."""
        return dict(self._registry)

    def _get_tool_aliases(self) -> dict[str, list[str]]:
        """Build alias map (canonical -> aliases) from registered tools.

        Only includes tools that declare aliases via Tool.tool_aliases.
        """
        if self._tool_aliases_cache is not None:
            return self._tool_aliases_cache

        mapping: dict[str, list[str]] = {}
        for tool_class in self._registry.values():
            # tool_name is guaranteed to exist by Tool.__init_subclass__
            aliases = tool_class.tool_aliases
            if aliases:
                mapping[tool_class.tool_name] = list(aliases)  # type: ignore[index]

        self._tool_aliases_cache = mapping
        return mapping

    def _get_tool_name_map(self) -> dict[str, str]:
        """Map of every supported tool name and alias -> canonical name.

        Built from Tool.tool_name and Tool.tool_aliases on all registered tools.
        """
        if self._tool_name_map_cache is not None:
            return self._tool_name_map_cache

        mapping: dict[str, str] = {}

        # Build mapping from alias map to avoid duplication
        alias_map = self._get_tool_aliases()

        # Add canonical names pointing to themselves
        for tool_name in self._registry.keys():
            mapping[tool_name] = tool_name

        # Add aliases pointing to canonical names
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                mapping[alias] = canonical

        self._tool_name_map_cache = mapping
        return mapping

    def clear(self) -> None:
        self._registry.clear()
        # Invalidate caches when registry changes
        self._tool_aliases_cache = None
        self._tool_name_map_cache = None

    def get_tool_display_names(self) -> list[str]:
        """Get friendly display names for all tools (canonical/alias1/alias2 format)."""
        alias_map = self._get_tool_aliases()
        display_names = []
        for canonical, alias_list in sorted(alias_map.items()):
            display_names.append("/".join([canonical] + alias_list) if alias_list else canonical)
        return display_names

    def get_canonical_name(self, tool_name: str) -> str | None:
        """Get canonical tool name from any name or alias.

        Args:
            tool_name: Tool name or alias (e.g., 'train', 'eval', 'simulate')

        Returns:
            Canonical tool name, or None if not found
        """
        name_map = self._get_tool_name_map()
        return name_map.get(tool_name)


# Global singleton instance
_registry = ToolRegistry()

# Register all tools
_registry.register(TrainTool)
_registry.register(EvalTool)
_registry.register(EvalRemoteTool)
_registry.register(PlayTool)
_registry.register(ReplayTool)
_registry.register(AnalysisTool)
_registry.register(SweepTool)


def get_tool_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton instance."""
    return _registry
