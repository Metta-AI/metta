"""Infer Tool factory from a recipe module and a verb when missing.

This provides a graceful fallback so users can run commands like
  - arena.play, arena.replay, arena.train, arena.sim, arena.eval
even if the recipe module does not explicitly define those functions,
as long as the module defines a function:

    def mettagrid() -> MettaGridConfig

The inferred tools are constructed with minimal required fields. Users can
still pass CLI overrides (e.g., policy_uris=...) which will be applied by the
runner before invocation.
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable, Optional

from metta.cogworks.curriculum import env_curriculum
from metta.rl.training import EvaluatorConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

VERB_ALIASES = {
    "eval": "evaluate",
    "evaluate": "evaluate",
    "train": "train",
    "play": "play",
    "replay": "replay",
    "sim": "sim",
}


def _resolve_mettagrid(module: ModuleType) -> MettaGridConfig | None:
    """Find and call mettagrid() in the recipe module to get a MettaGridConfig."""
    mg_fn = getattr(module, "mettagrid", None)
    if mg_fn is None or not callable(mg_fn):
        return None
    mg = mg_fn()
    if isinstance(mg, MettaGridConfig):
        return mg
    return None


def _make_default_names(module: ModuleType) -> tuple[str, str]:
    """Derive default (suite, name) from a module.

    Use 'eval' as the default simulation name to match existing recipes.
    """
    base = module.__name__.split(".")[-1]
    return base, "eval"


def _mettagrid_to_simulation(module: ModuleType, mg: MettaGridConfig) -> SimulationConfig:
    suite, name = _make_default_names(module)
    return SimulationConfig(suite=suite, name=name, env=mg)


def _mettagrid_to_training_env(mg: MettaGridConfig) -> TrainingEnvironmentConfig:
    return TrainingEnvironmentConfig(curriculum=env_curriculum(mg))


def _resolve_simulations(module: ModuleType) -> list[SimulationConfig] | None:
    """Find and call simulations() in the recipe module to get a list of SimulationConfig.

    If present and callable, it should return a sequence of SimulationConfig; we coerce to list.
    """
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


def try_infer_tool_factory(module: ModuleType, verb: str) -> Optional[Callable[[], object]]:
    """Return a zero-arg factory that creates a Tool inferred from the recipe module.

    - Returns None if inference is not possible.
    - Supports verbs: train, play, replay, sim, evaluate (alias: eval)
    - Requires the module to expose a MettaGrid configuration via the documented attributes.
    """
    normalized = VERB_ALIASES.get(verb, verb)
    if normalized not in {"train", "play", "replay", "sim", "evaluate"}:
        return None

    sims = _resolve_simulations(module)
    mg = _resolve_mettagrid(module)
    if sims is None and mg is None:
        return None

    # Build a zero-arg factory function so the runner can still apply overrides after construction
    if normalized == "train":
        # For training inference, require mettagrid() for a clear environment definition.
        if mg is None:
            return None

        def factory() -> TrainTool:
            kwargs = {}
            if sims is not None:
                kwargs["evaluator"] = EvaluatorConfig(simulations=sims)
            return TrainTool(training_env=_mettagrid_to_training_env(mg), **kwargs)

        return factory

    if normalized in {"play", "replay"}:

        def factory() -> PlayTool | ReplayTool:
            sim_cfg = _mettagrid_to_simulation(module, mg)
            if normalized == "play":
                return PlayTool(sim=sim_cfg)
            else:
                return ReplayTool(sim=sim_cfg)

        return factory

    if normalized in {"sim", "evaluate"}:

        def factory() -> EvalTool:
            if sims is not None:
                return EvalTool(simulations=sims)
            assert mg is not None
            sim_cfg = _mettagrid_to_simulation(module, mg)
            return EvalTool(simulations=[sim_cfg])

        return factory

    if normalized in {"evaluate_remote", "eval_remote", "sim_remote"}:

        def factory() -> EvalRemoteTool:
            if sims is not None:
                return EvalRemoteTool(simulations=sims)
            assert mg is not None
            sim_cfg = _mettagrid_to_simulation(module, mg)
            # EvalRemoteTool expects simulations list
            return EvalRemoteTool(simulations=[sim_cfg])

        return factory

    return None
