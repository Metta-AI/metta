"""Tool registry and discovery system.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class provides public methods:
- get_all_tools() -> All registered tool classes
- get_tool_display_names() -> Formatted display names (canonical/alias1/alias2)
- infer_tool_from_recipe() -> Attempt to infer tool from recipe module
- resolve_and_load_tool() -> Resolve user input and load tool maker
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.recipe import Recipe
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

    def _resolve_tool_path(self, tool_path: str) -> list[str]:
        """Convert tool path to candidate import paths.

        Args:
            tool_path: Tool path like 'arena.train', 'train', or 'eval'

        Examples:
            'arena.train' → ['arena.train', 'experiments.recipes.arena.train']
            'eval' → ['eval', 'evaluate'] (alias expansion)

        Returns ordered list of candidates to try.
        """
        candidates = [tool_path]

        # Add prefix for short forms (anything not starting with experiments.recipes.)
        if not tool_path.startswith("experiments.recipes."):
            candidates.append(f"experiments.recipes.{tool_path}")

        # Expand aliases if last component is a tool alias
        if "." in tool_path:
            module, verb = tool_path.rsplit(".", 1)
            alias_map = self._get_tool_aliases()

            # Check if verb is canonical and has aliases
            if verb in alias_map:
                for alias in alias_map[verb]:
                    candidates.append(f"{module}.{alias}")
                    if not module.startswith("experiments.recipes."):
                        candidates.append(f"experiments.recipes.{module}.{alias}")

            # Check if verb is an alias for a canonical name
            name_map = self._get_tool_name_map()
            if verb in name_map and name_map[verb] != verb:
                canonical = name_map[verb]
                candidates.append(f"{module}.{canonical}")
                if not module.startswith("experiments.recipes."):
                    candidates.append(f"experiments.recipes.{module}.{canonical}")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(candidates))

    def infer_tool_from_recipe(self, module_path: str, tool_name: str) -> Optional[Callable[[], Tool]]:
        """Try to infer a tool from a recipe module.

        Args:
            module_path: Recipe module like 'experiments.recipes.arena'
            tool_name: Tool to infer like 'train' or 'evaluate'

        Returns:
            Tool factory if inference succeeds, None otherwise
        """
        # Normalize to canonical tool name
        canonical = self._get_tool_name_map().get(tool_name, tool_name)

        # Get tool class
        tool_class = self._registry.get(canonical)
        if not tool_class:
            return None

        # Load recipe and try inference
        recipe = Recipe.load(module_path)
        if not recipe:
            return None

        return recipe.infer_tool(tool_class)

    def resolve_and_load_tool(self, tool_path: str) -> Callable[[], Tool] | None:
        """Resolve tool path and load the tool maker.

        Args:
            tool_path: Tool path like 'arena.train' or 'eval'

        Tries in order:
        1. Direct load (explicit tools defined in modules)
        2. Recipe inference (for recipes with mettagrid/simulations)

        Returns:
            Tool maker callable, or None if not found
        """
        candidates = self._resolve_tool_path(tool_path)

        for candidate in candidates:
            # Try direct load
            maker = _load_tool_maker(candidate)
            if maker:
                return maker

            # Try recipe inference
            if "." in candidate:
                module_path, tool_name = candidate.rsplit(".", 1)
                maker = self.infer_tool_from_recipe(module_path, tool_name)
                if maker:
                    return maker

        return None


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


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------


def _load_tool_maker(path: str) -> Optional[Callable[[], Tool]]:
    """Load a tool maker from an import path.

    Args:
        path: Import path like 'experiments.recipes.arena.train'

    Returns:
        Callable that creates a Tool, or None if not found
    """
    if "." not in path:
        return None

    module_path, symbol = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        maker = getattr(module, symbol, None)

        if maker is None:
            return None

        # Must be callable and return a Tool
        if not callable(maker):
            return None

        return maker  # type: ignore

    except (ImportError, AttributeError):
        return None
