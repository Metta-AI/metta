"""Path utilities and tool loading for the tool system."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool


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


def _resolve_tool_path(tool_path: str) -> list[str]:
    """Convert tool path to candidate import paths.

    Args:
        tool_path: Tool path like 'arena.train', 'train', or 'eval'

    Examples:
        'arena.train' → ['arena.train', 'experiments.recipes.arena.train']
        'eval' → ['eval', 'evaluate'] (alias expansion)

    Returns ordered list of candidates to try.
    """
    from metta.common.tool.tool_registry import get_tool_registry

    candidates = [tool_path]
    registry = get_tool_registry()

    # Add prefix for short forms (anything not starting with experiments.recipes.)
    if not tool_path.startswith("experiments.recipes."):
        candidates.append(f"experiments.recipes.{tool_path}")

    # Expand aliases if last component is a tool alias
    if "." in tool_path:
        module, verb = tool_path.rsplit(".", 1)

        # Get canonical name if this is an alias
        canonical = registry.get_canonical_name(verb)
        if canonical and canonical != verb:
            # Add version with canonical name
            candidates.append(f"{module}.{canonical}")
            if not module.startswith("experiments.recipes."):
                candidates.append(f"experiments.recipes.{module}.{canonical}")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))


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
        tool_path: Tool path like 'arena.train' or 'eval'

    Tries in order:
    1. Direct load (explicit tools defined in modules)
    2. Recipe inference (for recipes with mettagrid/simulations)

    Returns:
        Tool maker callable, or None if not found
    """
    from metta.common.tool.recipe import infer_tool_from_recipe

    candidates = _resolve_tool_path(tool_path)

    for candidate in candidates:
        # Try direct load
        maker = _load_tool_maker(candidate)
        if maker:
            return maker

        # Try recipe inference
        if "." in candidate:
            module_path, tool_name = candidate.rsplit(".", 1)
            maker = infer_tool_from_recipe(module_path, tool_name)
            if maker:
                return maker

    return None
