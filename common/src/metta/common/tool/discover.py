from __future__ import annotations

import inspect
from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

from metta.cogworks.curriculum import env_curriculum
from metta.common.tool import Tool
from metta.rl.training import EvaluatorConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

# -----------------------------------------------------------------------------
# Alias expansion (CLI-level) and candidate resolution helpers
# -----------------------------------------------------------------------------


def get_tool_aliases() -> dict[str, list[str]]:
    """Build default alias map (canonical -> aliases) from Tool classes.

    Only includes tools that declare aliases via Tool.tool_aliases.
    """
    mapping: dict[str, list[str]] = {}
    for cls in (EvalTool, EvalRemoteTool, TrainTool, PlayTool, ReplayTool):
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

    bases: list[str] = []
    if second:
        # Avoid redundant two-token cases like "train train"
        bases.append(f"{second}.{primary}")
    bases.append(primary)

    prefixes = auto_prefixes or []
    candidates: list[str] = []
    for base in bases:
        expanded: list[str] = [base]
        # Expand aliases if any; keep the canonical base first
        if "." in base:
            module_name, verb = base.rsplit(".", 1)
            alias_map = verb_aliases or get_tool_aliases()
            aliases = alias_map.get(verb, [])
            for v in aliases:
                expanded.append(f"{module_name}.{v}")
        for item in expanded:
            candidates.append(item)
            # Optionally try prefixed variants
            if (not short_only) or (item.count(".") <= 1):
                for pref in prefixes:
                    if not item.startswith(pref + "."):
                        candidates.append(f"{pref}.{item}")

    # Deduplicate preserving order
    ordered: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# -----------------------------------------------------------------------------
# Inference: Build tools from recipe modules exposing mettagrid()/simulations()
# -----------------------------------------------------------------------------


def get_available_tools(module: ModuleType) -> list[tuple[str, Callable[[], object]]]:
    """Return explicit tools (name, maker) defined in the module.

    - Includes Tool subclasses exported by the module and functions whose return
      annotation is a Tool subclass. The displayed name prefers the Tool class's
      `tool_name` if set; otherwise the attribute name is used.
    - Does NOT include inferred tools; inference is only used at execution time.
    """
    tools: dict[str, Callable[[], object]] = {}

    def register(name: str, maker: Callable[[], object]) -> None:
        if name not in tools:
            tools[name] = maker

    for name in dir(module):
        if name.startswith("_"):
            continue
        maker = getattr(module, name, None)
        if not callable(maker):
            continue
        # Class-based tool
        if inspect.isclass(maker) and issubclass(maker, Tool) and maker is not Tool:
            declared_name = getattr(maker, "tool_name", None)
            register((declared_name or name), maker)  # type: ignore[arg-type]
            continue
        # Function with return annotation of a Tool subclass
        if inspect.isfunction(maker):
            ret = inspect.signature(maker).return_annotation
            try:
                if inspect.isclass(ret) and issubclass(ret, Tool) and ret is not Tool:
                    declared_name = getattr(ret, "tool_name", None)
                    register((declared_name or name), maker)
                    continue
            except Exception:
                pass
            # Skip unannotated functions to avoid guessing; they remain runnable via CLI

    return sorted(tools.items(), key=lambda kv: kv[0])


def _resolve_mettagrid(module: ModuleType) -> MettaGridConfig | None:
    mg_fn = getattr(module, "mettagrid", None)
    if mg_fn is None or not callable(mg_fn):
        return None
    mg = mg_fn()
    if isinstance(mg, MettaGridConfig):
        return mg
    return None


def _make_default_names(module: ModuleType) -> tuple[str, str]:
    base = module.__name__.split(".")[-1]
    return base, "eval"


def _mettagrid_to_simulation(module: ModuleType, mg: MettaGridConfig) -> SimulationConfig:
    suite, name = _make_default_names(module)
    return SimulationConfig(suite=suite, name=name, env=mg)


def _mettagrid_to_training_env(mg: MettaGridConfig) -> TrainingEnvironmentConfig:
    return TrainingEnvironmentConfig(curriculum=env_curriculum(mg))


def _resolve_simulations(module: ModuleType) -> list[SimulationConfig] | None:
    fn = getattr(module, "simulations", None)
    if fn is None or not callable(fn):
        return None
    try:
        result = fn()
    except Exception:
        return None
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], SimulationConfig):
        return list(result)
    if isinstance(result, (list, tuple)) and not result:
        return []
    return None


def _supported_tool_classes() -> list[type[Tool]]:
    return [TrainTool, PlayTool, ReplayTool, EvalTool, EvalRemoteTool]


@lru_cache(maxsize=1)
def get_tool_name_map() -> dict[str, str]:
    """Map of every supported tool name and alias -> canonical name.

    Built from Tool.tool_name and Tool.tool_aliases on supported tool classes.
    Cached since this never changes at runtime.
    """
    mapping: dict[str, str] = {}
    for cls in _supported_tool_classes():
        tool_name = getattr(cls, "tool_name", None)
        if not tool_name:
            continue
        mapping[tool_name] = tool_name
        for alias in getattr(cls, "tool_aliases", []) or []:
            mapping[alias] = tool_name
    return mapping


def _resolve_canonical_tool_name(name: str) -> str | None:
    return get_tool_name_map().get(name)


def try_infer_tool_factory(module: ModuleType, verb: str) -> Optional[Callable[[], object]]:
    """Return a zero-arg factory that creates a Tool inferred from the recipe module.

    - Supports known tool classes and their aliases via `tool_name`/`tool_aliases`.
    - Returns None if inference is not possible for the requested tool.
    """
    normalized = _resolve_canonical_tool_name(verb) or verb
    inferable = set(get_tool_name_map().values())
    if normalized not in inferable:
        return None

    sims = _resolve_simulations(module)
    mg = _resolve_mettagrid(module)
    if sims is None and mg is None:
        return None

    if normalized == "train":
        if mg is None:
            return None

        def factory() -> TrainTool:
            kwargs = {}
            if sims is not None:
                kwargs["evaluator"] = EvaluatorConfig(simulations=sims)
            return TrainTool(training_env=_mettagrid_to_training_env(mg), **kwargs)

        return factory

    if normalized == "play":

        def factory() -> PlayTool:
            # Prefer simulations()[0] if available; otherwise fall back to mettagrid()
            if sims is not None and len(sims) > 0:
                sim_cfg = sims[0]
            else:
                assert mg is not None, "Cannot infer play: no simulations() provided and mettagrid() missing"
                sim_cfg = _mettagrid_to_simulation(module, mg)
            return PlayTool(sim=sim_cfg)

        return factory

    if normalized == "replay":

        def factory() -> ReplayTool:
            if sims is not None and len(sims) > 0:
                sim_cfg = sims[0]
            else:
                assert mg is not None, "Cannot infer replay: no simulations() provided and mettagrid() missing"
                sim_cfg = _mettagrid_to_simulation(module, mg)
            return ReplayTool(sim=sim_cfg)

        return factory

    if normalized == "evaluate":

        def factory() -> EvalTool:
            if sims is not None:
                return EvalTool(simulations=sims)
            assert mg is not None
            sim_cfg = _mettagrid_to_simulation(module, mg)
            return EvalTool(simulations=[sim_cfg])

        return factory

    if normalized == "evaluate_remote":

        def factory() -> EvalRemoteTool:
            if sims is not None:
                return EvalRemoteTool(simulations=sims)
            assert mg is not None
            sim_cfg = _mettagrid_to_simulation(module, mg)
            return EvalRemoteTool(simulations=[sim_cfg])

        return factory

    return None
