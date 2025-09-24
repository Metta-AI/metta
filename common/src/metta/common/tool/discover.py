from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from metta.cogworks.curriculum import env_curriculum
from metta.common.tool import Tool
from metta.rl.training import EvaluatorConfig
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
# Constants and Tool Class Registry
# -----------------------------------------------------------------------------

# These tools can be automatically inferred from a recipe's `mettagrid` or `simulations` functions
FACTORY_TOOL_CLASSES = (TrainTool, PlayTool, ReplayTool, EvalTool, EvalRemoteTool)
# This prefix can be omitted from the tool name in the CLI
DEFAULT_RECIPE_PREFIX = "experiments.recipes"

# -----------------------------------------------------------------------------
# Alias expansion (CLI-level) and candidate resolution helpers
# -----------------------------------------------------------------------------


def get_tool_aliases() -> dict[str, list[str]]:
    """Build default alias map (canonical -> aliases) from Tool classes.

    Only includes tools that declare aliases via Tool.tool_aliases.
    """
    mapping: dict[str, list[str]] = {}
    for cls in FACTORY_TOOL_CLASSES:
        name = getattr(cls, "tool_name", None)
        aliases = getattr(cls, "tool_aliases", [])
        if name and aliases:
            mapping[name] = list(aliases)
    return mapping


def generate_candidate_paths(
    primary: str | None,
    second: str | None = None,
    *,
    auto_prefixes: list[str] | None = None,
    short_only: bool = True,
    verb_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Generate ordered candidate import paths.

    - primary: main symbol path like "arena.train" or fully-qualified.
    - second: when provided, treats inputs like (x, y) as the sugar y.x.
    - auto_prefixes: optional module prefixes to try (e.g., ["experiments.recipes"]).
    - short_only: if True, only apply prefixes for short forms (<= 1 dot).
    """
    if not primary:
        return []

    # Build base paths from primary and optional second token
    bases: list[str] = []
    if second:
        bases.append(f"{second}.{primary}")
    bases.append(primary)

    # Expand with aliases and prefixes
    candidates: list[str] = []
    alias_map = verb_aliases or get_tool_aliases()
    prefixes = auto_prefixes or []

    for base in bases:
        # Start with base and expand with verb aliases if present
        expanded: list[str] = [base]
        if "." in base:
            module_name, verb = base.rsplit(".", 1)
            for alias in alias_map.get(verb, []):
                expanded.append(f"{module_name}.{alias}")

        # Add each expansion with optional prefixes
        for item in expanded:
            candidates.append(item)
            # Apply prefixes for short forms
            if (not short_only) or (item.count(".") <= 1):
                for pref in prefixes:
                    if not item.startswith(pref + "."):
                        candidates.append(f"{pref}.{item}")

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


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


def _resolve_mettagrid(module: ModuleType) -> MettaGridConfig | None:
    """Safely call module.mettagrid() if it exists."""
    mg_fn = getattr(module, "mettagrid", None)
    if not callable(mg_fn):
        return None

    try:
        mg = mg_fn()
        return mg if isinstance(mg, MettaGridConfig) else None
    except Exception:
        return None


def _resolve_simulations(module: ModuleType) -> list[SimulationConfig] | None:
    """Safely call module.simulations() if it exists."""
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

    Built from Tool.tool_name and Tool.tool_aliases on ALL Tool subclasses.
    Cached since this never changes at runtime.
    """
    mapping: dict[str, str] = {}
    # Include factory tools that can be inferred
    for cls in FACTORY_TOOL_CLASSES:
        tool_name = getattr(cls, "tool_name", None)
        if not tool_name:
            continue
        mapping[tool_name] = tool_name
        for alias in getattr(cls, "tool_aliases", []) or []:
            mapping[alias] = tool_name

    # Also include other known tool types (could be extended)
    # These might not be in FACTORY_TOOL_CLASSES but still have tool_name
    for cls in (AnalysisTool, SweepTool):
        tool_name = getattr(cls, "tool_name", None)
        if tool_name and tool_name not in mapping:
            mapping[tool_name] = tool_name
            for alias in getattr(cls, "tool_aliases", []) or []:
                mapping[alias] = tool_name

    return mapping


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

    - Supports known tool classes and their aliases via `tool_name`/`tool_aliases`.
    - Returns None if inference is not possible for the requested tool.
    """
    # Normalize tool name to canonical form
    normalized = get_tool_name_map().get(verb, verb)
    inferable = set(get_tool_name_map().values())
    if normalized not in inferable:
        return None

    # Try to get recipe configurations
    sims = _resolve_simulations(module)
    mg = _resolve_mettagrid(module)
    if sims is None and mg is None:
        return None

    # Build factory based on tool type
    if normalized == "train":
        if mg is None:
            return None

        def train_factory() -> TrainTool:
            kwargs = {}
            if sims is not None:
                kwargs["evaluator"] = EvaluatorConfig(simulations=sims)
            return TrainTool(training_env=_mettagrid_to_training_env(mg), **kwargs)

        return train_factory

    if normalized == "play":

        def play_factory() -> PlayTool:
            # Prefer simulations()[0] if available; otherwise fall back to mettagrid()
            if sims and len(sims) > 0:
                sim_cfg = sims[0]
            elif mg is not None:
                sim_cfg = _mettagrid_to_simulation(module, mg)
            else:
                raise ValueError("Cannot infer play: no simulations() provided and mettagrid() missing")
            return PlayTool(sim=sim_cfg)

        return play_factory

    if normalized == "replay":

        def replay_factory() -> ReplayTool:
            # Prefer simulations()[0] if available; otherwise fall back to mettagrid()
            if sims and len(sims) > 0:
                sim_cfg = sims[0]
            elif mg is not None:
                sim_cfg = _mettagrid_to_simulation(module, mg)
            else:
                raise ValueError("Cannot infer replay: no simulations() provided and mettagrid() missing")
            return ReplayTool(sim=sim_cfg)

        return replay_factory

    if normalized == "evaluate":

        def eval_factory() -> EvalTool:
            if sims is not None:
                return EvalTool(simulations=sims)
            assert mg is not None, "Cannot infer evaluate: mettagrid() missing"
            sim_cfg = _mettagrid_to_simulation(module, mg)
            return EvalTool(simulations=[sim_cfg])

        return eval_factory

    if normalized == "evaluate_remote":

        def eval_remote_factory() -> EvalRemoteTool:
            if sims is not None:
                return EvalRemoteTool(simulations=sims)
            assert mg is not None, "Cannot infer evaluate_remote: mettagrid() missing"
            sim_cfg = _mettagrid_to_simulation(module, mg)
            return EvalRemoteTool(simulations=[sim_cfg])

        return eval_remote_factory

    return None
