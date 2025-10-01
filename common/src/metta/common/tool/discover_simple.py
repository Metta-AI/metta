"""Simplified tool discovery and path resolution.

Focuses on:
1. Resolving user input to importable paths (e.g., 'arena.train' → 'experiments.recipes.arena.train')
2. Loading tool makers (functions/classes that create Tools)
3. Auto-factory integration for recipes with mettagrid()/simulations()

Avoids unnecessary metaprogramming and complex introspection.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.common.tool.tool_registry import get_tool_registry
from metta.sim.simulation_config import SimulationConfig

# -----------------------------------------------------------------------------
# Tool Registration (explicit, not magical)
# -----------------------------------------------------------------------------
from metta.tools.analyze import AnalysisTool
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

_registry = get_tool_registry()
_registry.register(TrainTool)
_registry.register(EvalTool)
_registry.register(EvalRemoteTool)
_registry.register(PlayTool)
_registry.register(ReplayTool)
_registry.register(AnalysisTool)
_registry.register(SweepTool)


# -----------------------------------------------------------------------------
# Simple Path Resolution
# -----------------------------------------------------------------------------


def resolve_tool_path(user_input: str, second_token: str | None = None) -> list[str]:
    """Convert user input to candidate import paths.

    Examples:
        'arena.train' → ['arena.train', 'experiments.recipes.arena.train']
        'train' + 'arena' → ['arena.train', 'experiments.recipes.arena.train']
        'eval' → ['eval', 'evaluate'] (alias expansion)

    Returns ordered list of candidates to try.
    """
    # Build base path(s)
    if second_token:
        # Two-token form: 'train arena' → 'arena.train'
        base = f"{second_token}.{user_input}"
    else:
        base = user_input

    candidates = [base]

    # Add prefix for short forms (no dots or single dot)
    if base.count(".") <= 1:
        candidates.append(f"experiments.recipes.{base}")

    # Expand aliases if last component is a tool alias
    if "." in base:
        module, verb = base.rsplit(".", 1)
        alias_map = get_tool_registry().get_tool_aliases()

        # Check if verb is canonical and has aliases
        if verb in alias_map:
            for alias in alias_map[verb]:
                candidates.append(f"{module}.{alias}")
                if module.count(".") == 0:  # Short form
                    candidates.append(f"experiments.recipes.{module}.{alias}")

        # Check if verb is an alias for a canonical name
        name_map = get_tool_registry().get_tool_name_map()
        if verb in name_map and name_map[verb] != verb:
            canonical = name_map[verb]
            candidates.append(f"{module}.{canonical}")
            if module.count(".") == 0:
                candidates.append(f"experiments.recipes.{module}.{canonical}")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))


# -----------------------------------------------------------------------------
# Tool Loading (no complex introspection)
# -----------------------------------------------------------------------------


def load_tool_maker(path: str) -> Optional[Callable[[], Tool]]:
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


# -----------------------------------------------------------------------------
# Recipe Integration (simplified auto-factory)
# -----------------------------------------------------------------------------


@lru_cache(maxsize=128)
def get_recipe_configs(module: ModuleType) -> tuple[MettaGridConfig | None, list[SimulationConfig] | None]:
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


def infer_tool_from_recipe(module_path: str, tool_name: str) -> Optional[Callable[[], Tool]]:
    """Try to infer a tool from a recipe module using auto_factory.

    Args:
        module_path: Recipe module like 'experiments.recipes.arena'
        tool_name: Tool to infer like 'train' or 'evaluate'

    Returns:
        Tool factory if inference succeeds, None otherwise
    """
    # Normalize to canonical tool name
    name_map = get_tool_registry().get_tool_name_map()
    canonical = name_map.get(tool_name, tool_name)

    # Get tool class
    tool_class = get_tool_registry().get_all_tools().get(canonical)
    if not tool_class:
        return None

    # Skip if tool doesn't support auto_factory
    if tool_class.auto_factory == Tool.auto_factory:
        return None

    # Load recipe module
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None

    # Get configs
    mg, sims = get_recipe_configs(module)
    if mg is None and sims is None:
        return None

    # Try auto_factory
    tool = tool_class.auto_factory(mettagrid=mg, simulations=sims)
    if tool is None:
        return None

    # Return factory
    return lambda: tool


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def resolve_and_load_tool(
    user_input: str, second_token: str | None = None
) -> tuple[str | None, Callable[[], Tool] | None]:
    """Resolve user input and load the tool maker.

    Tries in order:
    1. Direct load (explicit tools defined in modules)
    2. Auto-factory inference (for recipes with mettagrid/simulations)

    Returns:
        (resolved_path, tool_maker) or (None, None) if not found
    """
    candidates = resolve_tool_path(user_input, second_token)

    for candidate in candidates:
        # Try direct load
        maker = load_tool_maker(candidate)
        if maker:
            return candidate, maker

        # Try auto-factory inference
        if "." in candidate:
            module_path, tool_name = candidate.rsplit(".", 1)
            maker = infer_tool_from_recipe(module_path, tool_name)
            if maker:
                return candidate, maker

    return None, None
