"""Path utilities and tool loading for the tool system."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.recipe_registry import recipe_registry


def validate_module_path(module_path: str) -> bool:
    """Validate that a module path can be resolved to a tool maker.

    Args:
        module_path: Module path like 'arena.train' or 'experiments.recipes.arena.train'

    Returns:
        True if the module path can be resolved, False otherwise
    """
    return resolve_and_load_tool_maker(module_path) is not None


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


def resolve_and_load_tool_maker(tool_path: str) -> Callable[[], Tool] | None:
    """Resolve tool path and load the tool maker.

    Args:
        tool_path: Tool path like 'arena.train' or 'experiments.recipes.arena.train'

    Resolution strategy:
    1. Try direct import (e.g., 'experiments.recipes.arena.train')
    2. If not found and path has a dot, split into module_path and tool_maker_name:
       - Look up recipe and get tool maker by name (handles both function names like
         'replay_null', 'train_shaped' and tool class names like 'evaluate', 'train')

    Returns:
        Tool maker callable, or None if not found
    """

    # Phase 1: Try direct import
    maker = _load_tool_maker(tool_path)
    if maker:
        return maker

    # Phase 2: Try recipe lookup
    if "." in tool_path:
        module_path, tool_name = tool_path.rsplit(".", 1)
        recipe = recipe_registry.get(module_path)
        if recipe:
            return recipe.get_tool_maker(tool_name)

    return None
