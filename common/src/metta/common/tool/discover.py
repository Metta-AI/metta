from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from metta.cogworks.curriculum import env_curriculum
from metta.common.tool import Tool
from metta.common.tool.tool_registry import get_tool_registry
from metta.rl.training.training_environment import TrainingEnvironmentConfig
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
# Tool Registration
# -----------------------------------------------------------------------------

# Register all tools explicitly
_registry = get_tool_registry()
_registry.register(TrainTool)
_registry.register(EvalTool)
_registry.register(EvalRemoteTool)
_registry.register(PlayTool)
_registry.register(ReplayTool)
_registry.register(AnalysisTool)
_registry.register(SweepTool)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# This prefix can be omitted from the tool name in the CLI
DEFAULT_RECIPE_PREFIX = "experiments.recipes"

# -----------------------------------------------------------------------------
# Alias expansion (CLI-level) and candidate resolution helpers
# -----------------------------------------------------------------------------


def get_tool_aliases() -> dict[str, list[str]]:
    """Build default alias map (canonical -> aliases) from registered tools.

    Only includes tools that declare aliases via Tool.tool_aliases.
    """
    return get_tool_registry().get_tool_aliases()


def generate_candidate_paths(
    primary: str | None,
    second: str | None = None,
) -> list[str]:
    """Generate ordered candidate import paths for metta tools.

    Automatically handles shorthand expansion:
    - Adds "experiments.recipes" prefix for short forms (<= 1 dot)
    - Expands tool aliases (e.g., train/t, evaluate/eval/sim)
    - Supports two-token form (e.g., "train arena" → "arena.train")

    Args:
        primary: main symbol path like "arena.train" or fully-qualified.
        second: when provided, treats inputs like (x, y) as the sugar y.x.
    """
    if not primary:
        return []

    # Build base paths from primary and optional second token
    bases: list[str] = []
    if second:
        bases.append(f"{second}.{primary}")
    bases.append(primary)

    # Expand with metta-specific aliases and prefixes
    candidates: list[str] = []
    alias_map = get_tool_aliases()
    prefixes = ["experiments.recipes"]

    for base in bases:
        # Start with base and expand with verb aliases if present
        expanded: list[str] = [base]
        if "." in base:
            module_name, verb = base.rsplit(".", 1)
            for alias in alias_map.get(verb, []):
                expanded.append(f"{module_name}.{alias}")

        # Add each expansion with prefixes for short forms
        for item in expanded:
            candidates.append(item)
            # Apply prefixes only for short forms (<= 1 dot)
            if item.count(".") <= 1:
                for pref in prefixes:
                    if not item.startswith(pref + "."):
                        candidates.append(f"{pref}.{item}")

    return list(dict.fromkeys(candidates))


# -----------------------------------------------------------------------------
# Inference: Build tools from recipe modules exposing mettagrid()/simulations()
# -----------------------------------------------------------------------------


def get_available_tools(module: ModuleType) -> list[tuple[str, Callable[[], object]]]:
    """Return explicit tools (name, maker) defined in the module.

    - Includes Tool subclasses exported by the module and functions whose return
      annotation is a Tool subclass. Display names use the attribute/function
      name (e.g., `train_shaped`) for clarity.
    - Does NOT include inferred tools; inference is only used at execution time.
    """
    tools: dict[str, Callable[[], object]] = {}

    for name in dir(module):
        if name.startswith("_"):
            continue

        maker = getattr(module, name, None)
        if not callable(maker):
            continue

        # Check if it's a Tool class defined in this module
        if (
            inspect.isclass(maker)
            and issubclass(maker, Tool)
            and maker is not Tool
            and getattr(maker, "__module__", None) == module.__name__
        ):
            if name not in tools:
                tools[name] = maker  # type: ignore[assignment]
            continue

        # Check if it's a function returning a Tool
        if inspect.isfunction(maker):
            try:
                ret = inspect.signature(maker).return_annotation
                if inspect.isclass(ret) and issubclass(ret, Tool) and ret is not Tool:
                    if name not in tools:
                        tools[name] = maker
            except Exception:
                pass

    return sorted(tools.items(), key=lambda kv: kv[0])


# -----------------------------------------------------------------------------
# Recipe Module Helpers
# -----------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _resolve_mettagrid(module: ModuleType) -> MettaGridConfig | None:
    """Safely call module.mettagrid() if it exists.

    Cached to avoid repeated construction when checking multiple tools in same module.
    """
    mg_fn = getattr(module, "mettagrid", None)
    if not callable(mg_fn):
        return None

    try:
        mg = mg_fn()
        return mg if isinstance(mg, MettaGridConfig) else None
    except Exception:
        return None


@lru_cache(maxsize=128)
def _resolve_simulations(module: ModuleType) -> list[SimulationConfig] | None:
    """Safely call module.simulations() if it exists.

    Cached to avoid repeated construction when checking multiple tools in same module.
    Performance optimization: simulations() can create hundreds of objects for eval suites.
    """
    fn = getattr(module, "simulations", None)
    if not callable(fn):
        return None

    try:
        result = fn()
        if isinstance(result, (list, tuple)):
            if not result:
                return []
            if isinstance(result[0], SimulationConfig):
                return list(result)
    except Exception:
        pass
    return None


def _make_default_names(module: ModuleType) -> tuple[str, str]:
    """Generate default suite and name for a module."""
    base = module.__name__.split(".")[-1]
    return base, "eval"


def _mettagrid_to_simulation(module: ModuleType, mg: MettaGridConfig) -> SimulationConfig:
    """Convert a MettaGridConfig to a SimulationConfig."""
    suite, name = _make_default_names(module)
    return SimulationConfig(suite=suite, name=name, env=mg)


def _mettagrid_to_training_env(mg: MettaGridConfig) -> TrainingEnvironmentConfig:
    """Convert a MettaGridConfig to a TrainingEnvironmentConfig."""
    return TrainingEnvironmentConfig(curriculum=env_curriculum(mg))


@lru_cache(maxsize=1)
def get_tool_name_map() -> dict[str, str]:
    """Map of every supported tool name and alias -> canonical name.

    Built from Tool.tool_name and Tool.tool_aliases on all registered tools.
    Cached since the registry doesn't change at runtime.
    """
    return get_tool_registry().get_tool_name_map()


def _function_returns_tool_type(func_maker: Callable, tool_type_name: str) -> bool:
    """Check if a function returns a specific Tool type.

    Args:
        func_maker: The function to check
        tool_type_name: The canonical tool name (e.g., 'evaluate', 'sweep')
    """
    if not inspect.isfunction(func_maker):
        return False

    try:
        sig = inspect.signature(func_maker)
        ret_annotation = sig.return_annotation

        # Check if return type is a Tool subclass
        if inspect.isclass(ret_annotation) and issubclass(ret_annotation, Tool):
            # Check if this Tool class has the matching tool_name
            return getattr(ret_annotation, "tool_name", None) == tool_type_name
    except Exception:
        pass

    return False


def list_recipes_supporting_tool(tool_name: str) -> list[str]:
    """Find all recipe modules that support a given tool.

    Returns a sorted list of full paths like 'experiments.recipes.arena.train'.
    Handles both inferred tools and explicitly defined tools.
    """
    # Normalize the tool name to canonical form if it's a known alias
    canonical = get_tool_name_map().get(tool_name, tool_name)

    supported: list[str] = []

    try:
        recipes_pkg = importlib.import_module("experiments.recipes")
    except Exception:
        return []

    if not hasattr(recipes_pkg, "__path__"):
        return []

    # Walk all recipe modules
    for _, module_name, _ in pkgutil.walk_packages(recipes_pkg.__path__, recipes_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        # Get all explicitly defined tools in this module
        explicit_tools = get_available_tools(mod)

        # Check explicit tools - look for both exact name matches AND tools that return the requested type
        for func_name, func_maker in explicit_tools:
            # Direct name match (e.g., 'sweep' matches 'sweep')
            if func_name == canonical:
                supported.append(f"{module_name}.{func_name}")
                continue

            # Check if this function returns the requested tool type
            # e.g., 'evaluate_in_sweep' might return an EvalTool
            if _function_returns_tool_type(func_maker, canonical):
                supported.append(f"{module_name}.{func_name}")

        # Also check if it can be inferred (for factory tools like train, play, etc.)
        # But only if we haven't already found it explicitly
        already_found = any(f"{module_name}.{canonical}" in s for s in supported)
        if not already_found:
            try:
                if try_infer_tool_factory(mod, canonical):
                    supported.append(f"{module_name}.{canonical}")
            except Exception:
                pass

    return sorted(set(supported))


def try_infer_tool_factory(module: ModuleType, verb: str) -> Optional[Callable[[], object]]:
    """Return a zero-arg factory that creates a Tool inferred from the recipe module.

    Delegates to the tool's auto_factory() class method.
    Returns None if the tool doesn't support auto-factory or can't be inferred.

    TODO: Move auto_factory logic to a Recipe base class. Instead of tools defining
    how to construct themselves from recipes, recipes should define how to construct
    each tool type. This would make recipes first-class and eliminate the need for
    tools to know about recipe structure.
    """
    # Normalize tool name to canonical form
    normalized = get_tool_name_map().get(verb, verb)

    # Get the tool class from the registry
    all_tools = get_tool_registry().get_all_tools()
    tool_class = all_tools.get(normalized)
    if tool_class is None:
        return None

    # Check if tool supports auto_factory before resolving configs (performance optimization)
    # If auto_factory is not overridden from Tool base class, skip expensive config resolution
    from metta.common.tool import Tool

    if tool_class.auto_factory == Tool.auto_factory:
        return None

    # Try to get recipe configurations (only if tool supports auto_factory)
    sims = _resolve_simulations(module)
    mg = _resolve_mettagrid(module)
    if sims is None and mg is None:
        return None

    # Call the tool's auto_factory method
    tool_instance = tool_class.auto_factory(mettagrid=mg, simulations=sims)
    if tool_instance is None:
        return None

    # Return a zero-arg factory that returns the tool instance
    return lambda: tool_instance
