"""Path utilities and tool loading for the tool system."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.tool_registry import get_tool_registry


def normalize_module_path(module_path: str) -> list[str]:
    """Normalize a module path to possible full paths.

    Args:
        module_path: Module path like 'arena' or 'experiments.recipes.arena'

    Returns:
        List of candidate full module paths to try (e.g., ['arena', 'experiments.recipes.arena'])
    """
    candidates = [module_path]
    if not module_path.startswith("experiments.recipes."):
        candidates.append(f"experiments.recipes.{module_path}")
    return candidates


def strip_recipe_prefix(module_path: str) -> str:
    """Strip the 'experiments.recipes.' prefix if present.

    Args:
        module_path: Full module path like 'experiments.recipes.arena'

    Returns:
        Short name like 'arena'
    """
    return module_path.replace("experiments.recipes.", "")


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

        # Get canonical tool name (handle aliases)
        registry = get_tool_registry()
        canonical = registry.get_canonical_name(tool_name) or tool_name

        # Try both short and full recipe paths
        recipe_registry = get_recipe_registry()
        for candidate_module in normalize_module_path(module_path):
            recipe = recipe_registry.get(candidate_module)
            if recipe:
                # Found recipe! Try to get tool
                tools = recipe.get_tools_for_canonical(canonical)
                if tools:
                    # Return first matching tool
                    return tools[0][1]

    return None
