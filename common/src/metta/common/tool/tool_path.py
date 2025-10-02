"""Path utilities and tool loading for the tool system."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.tool_registry import get_tool_registry


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


def resolve_and_load_tool(tool_path: str) -> Callable[[], Tool] | None:
    """Resolve tool path and load the tool maker.

    Args:
        tool_path: Tool path like 'arena.train', 'train arena', or 'eval'

    Two-phase resolution:
    1. Try raw path - direct load from import path
    2. Try recipe registry - check if path maps to a known recipe, then infer tool

    Returns:
        Tool maker callable, or None if not found
    """
    from metta.common.tool.recipe_registry import get_recipe_registry

    # Phase 1: Try direct load with raw path
    maker = _load_tool_maker(tool_path)
    if maker:
        return maker

    # Phase 2: Check if this maps to a recipe in the registry
    # Parse tool_path into possible (module_path, tool_name) combinations
    if "." in tool_path:
        module_path, tool_name = tool_path.rsplit(".", 1)

        # Check recipe registry (handles both short and full paths)
        recipe_registry = get_recipe_registry()
        recipe = recipe_registry.get(module_path)
        if recipe:
            # First try direct name lookup (for functions like train_shaped, replay_null)
            if tool_name in recipe.get_all_tool_names():
                # Access the tool directly from the _name_to_tool map
                return recipe._name_to_tool.get(tool_name)

            # If direct lookup fails, try canonical name (for aliases and inferred tools)
            registry = get_tool_registry()
            canonical = registry.get_canonical_name(tool_name) or tool_name
            tools = recipe.get_tools_for_canonical(canonical)
            if tools:
                # Return first matching tool
                return tools[0][1]

    return None
