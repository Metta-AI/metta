"""Tool registry and discovery system.

Combines:
- ToolRegistry: Central registry of all tools
- Path resolution: Convert user input to import paths
- Tool loading: Load and instantiate tools from paths
- Recipe integration: Auto-factory support for recipes

This module replaces the split between tool_registry.py and discover.py.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from metta.common.tool import Tool
from metta.sim.simulation_config import SimulationConfig
from metta.tools.analyze import AnalysisTool
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Singleton registry for all tools in the system.

    Tools are explicitly registered on module import (see registration section below).
    """

    _instance: ToolRegistry | None = None
    _registry: dict[str, type[Tool]]

    def __new__(cls) -> ToolRegistry:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, tool_class: type[Tool]) -> None:
        """Register a tool class.

        Args:
            tool_class: The tool class to register (must have tool_name set)
        """
        # tool_name is guaranteed to exist by Tool.__init_subclass__ validation
        tool_name = tool_class.tool_name
        assert tool_name is not None, f"Tool class {tool_class.__name__} has no tool_name (validation failed?)"

        if tool_name in self._registry:
            # Allow re-registration of the same class (e.g., during testing or reloading)
            existing = self._registry[tool_name]
            if existing is not tool_class:
                raise ValueError(
                    f"Tool name '{tool_name}' already registered by {existing.__name__}, "
                    f"cannot register {tool_class.__name__}"
                )
        self._registry[tool_name] = tool_class

    def get_all_tools(self) -> dict[str, type[Tool]]:
        """Get all registered tool classes mapped by their canonical name."""
        return dict(self._registry)

    def get_tool_aliases(self) -> dict[str, list[str]]:
        """Build alias map (canonical -> aliases) from registered tools.

        Only includes tools that declare aliases via Tool.tool_aliases.
        """
        mapping: dict[str, list[str]] = {}
        for tool_class in self._registry.values():
            # tool_name is guaranteed to exist by Tool.__init_subclass__
            aliases = tool_class.tool_aliases
            if aliases:
                mapping[tool_class.tool_name] = list(aliases)  # type: ignore[index]
        return mapping

    def get_tool_name_map(self) -> dict[str, str]:
        """Map of every supported tool name and alias -> canonical name.

        Built from Tool.tool_name and Tool.tool_aliases on all registered tools.
        """
        mapping: dict[str, str] = {}

        # Build mapping from alias map to avoid duplication
        alias_map = self.get_tool_aliases()

        # Add canonical names pointing to themselves
        for tool_name in self._registry.keys():
            mapping[tool_name] = tool_name

        # Add aliases pointing to canonical names
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                mapping[alias] = canonical

        return mapping

    def clear(self) -> None:
        """Clear the registry. Mainly for testing."""
        self._registry.clear()


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
# Path Resolution
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_tool_name_map() -> dict[str, str]:
    """Cached access to tool name map (canonical and aliases → canonical)."""
    return _registry.get_tool_name_map()


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
        alias_map = _registry.get_tool_aliases()

        # Check if verb is canonical and has aliases
        if verb in alias_map:
            for alias in alias_map[verb]:
                candidates.append(f"{module}.{alias}")
                if module.count(".") == 0:  # Short form
                    candidates.append(f"experiments.recipes.{module}.{alias}")

        # Check if verb is an alias for a canonical name
        name_map = get_tool_name_map()
        if verb in name_map and name_map[verb] != verb:
            canonical = name_map[verb]
            candidates.append(f"{module}.{canonical}")
            if module.count(".") == 0:
                candidates.append(f"experiments.recipes.{module}.{canonical}")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))


# -----------------------------------------------------------------------------
# Tool Loading
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
# Recipe Integration (Auto-Factory)
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
    """Try to infer a tool from a recipe module.

    Args:
        module_path: Recipe module like 'experiments.recipes.arena'
        tool_name: Tool to infer like 'train' or 'evaluate'

    Returns:
        Tool factory if inference succeeds, None otherwise
    """
    from metta.common.tool import Tool

    # Normalize to canonical tool name
    name_map = get_tool_name_map()
    canonical = name_map.get(tool_name, tool_name)

    # Get tool class
    tool_class = _registry.get_all_tools().get(canonical)
    if not tool_class:
        return None

    # Skip if tool doesn't support inference
    if tool_class.infer == Tool.infer:
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

    # Try inferring the tool
    tool = tool_class.infer(mettagrid=mg, simulations=sims)
    if tool is None:
        return None

    # Return factory
    return lambda: tool


def get_available_tools(module: ModuleType) -> list[tuple[str, Callable[[], object]]]:
    """Get explicit tool-returning functions defined in a module.

    Uses type hints to identify functions that return Tool instances.
    """
    from typing import get_type_hints

    from metta.common.tool import Tool

    tools: list[tuple[str, Callable[[], object]]] = []

    for name in dir(module):
        # Skip private/special attributes
        if name.startswith("_"):
            continue

        attr = getattr(module, name)

        # Must be callable and not a class
        if not callable(attr) or isinstance(attr, type):
            continue

        # Check return type hint
        try:
            hints = get_type_hints(attr)
            return_type = hints.get("return")

            # Check if return type is a Tool subclass
            if return_type and isinstance(return_type, type) and issubclass(return_type, Tool):
                tools.append((name, attr))
        except Exception:
            # No type hints or invalid hints - skip
            pass

    return tools


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
