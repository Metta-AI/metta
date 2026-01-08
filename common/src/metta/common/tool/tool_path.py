"""Path utilities and tool loading for the tool system."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.recipe_registry import recipe_registry


def validate_module_path(module_path: str) -> bool:
    """Validate that a module path can be resolved to a tool maker.

    Args:
        module_path: Module path like 'arena.train' or 'recipes.experiment.arena.train'

    Returns:
        True if the module path can be resolved, False otherwise
    """
    return resolve_and_load_tool_maker(module_path) is not None


def _load_tool_maker(path: str) -> Optional[Callable[[], Tool]]:
    """Load a tool maker from an import path.

    Args:
        path: Import path like 'recipes.experiment.arena.train'

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


def parse_two_token_syntax(first_token: str, second_token: str | None) -> tuple[str, int]:
    """Parse two-token tool syntax like 'train arena' into 'arena.train'.

    Args:
        first_token: First token (e.g., 'train' or 'arena.train')
        second_token: Optional second token (e.g., 'arena')

    Returns:
        Tuple of (resolved_path, args_consumed) where:
        - resolved_path: The final module path to use
        - args_consumed: Number of additional arguments consumed (0 or 1)

    Examples:
        >>> parse_two_token_syntax('arena.train', None)
        ('arena.train', 0)
        >>> parse_two_token_syntax('train', 'arena')
        ('arena.train', 1)
        >>> parse_two_token_syntax('train', 'run=test')  # Second token is not a module
        ('train', 0)
    """
    # If second_token looks like an argument (contains = or starts with -), don't use it
    if second_token and ("=" in second_token or second_token.startswith("-")):
        return (first_token, 0)

    # Try two-token form if second_token is provided
    if second_token:
        two_token_path = f"{second_token}.{first_token}"
        # Check if this resolves to a valid tool maker
        if resolve_and_load_tool_maker(two_token_path):
            return (two_token_path, 1)

    # Fall back to first token only
    return (first_token, 0)


def resolve_and_load_tool_maker(tool_path: str) -> Callable[[], Tool] | None:
    """Resolve tool path and load the tool maker.

    Args:
        tool_path: Tool path like 'arena.train' or 'recipes.experiment.arena.train'

    Resolution strategy:
    1. Try direct import (e.g., 'recipes.experiment.arena.train')
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
