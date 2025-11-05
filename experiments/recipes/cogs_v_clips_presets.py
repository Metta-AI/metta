"""Utilities for building CoGs vs Clips recipe presets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Callable, Optional, Sequence, TypeVar
from typing_extensions import ParamSpec

from experiments.recipes.cogs_v_clips import (
    evaluate as base_evaluate,
    make_curriculum as base_make_curriculum,
    make_eval_suite as base_make_eval_suite,
    make_training_env as base_make_training_env,
    play as base_play,
    play_training_env as base_play_training_env,
    train as base_train,
    train_coordination as base_train_coordination,
    train_large_maps as base_train_large_maps,
    train_medium_maps as base_train_medium_maps,
    train_single_mission as base_train_single_mission,
    train_small_maps as base_train_small_maps,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class CvcPreset:
    """Collection of callable wrappers bound to default variant presets."""

    default_variants: tuple[str, ...]
    make_eval_suite: Callable[
        [int, str | None, Optional[Sequence[str]], Optional[Sequence[str]]],
        list[SimulationConfig],
    ]
    make_training_env: Callable[[int, str, Optional[Sequence[str]]], MettaGridConfig]
    make_curriculum: Callable[
        [
            int,
            Optional[list[str]],
            bool,
            Optional[CurriculumAlgorithmConfig],
            Optional[Sequence[str]],
        ],
        CurriculumConfig,
    ]
    train: Callable[
        [
            int,
            Optional[CurriculumConfig],
            Optional[list[str]],
            bool,
            Optional[Sequence[str]],
            Optional[Sequence[str]],
            str | None,
        ],
        TrainTool,
    ]
    train_small_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_medium_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_large_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_coordination: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_single_mission: Callable[
        [str, int, Optional[Sequence[str]], Optional[Sequence[str]], str | None],
        TrainTool,
    ]
    evaluate: Callable[
        [
            str | Sequence[str] | None,
            int,
            str | None,
            Optional[Sequence[str]],
            Optional[Sequence[str]],
        ],
        EvaluateTool,
    ]
    play: Callable[[Optional[str], str, int, Optional[Sequence[str]]], PlayTool]
    play_training_env: Callable[[Optional[str], int, Optional[Sequence[str]]], PlayTool]


def _wrap_with_default_variants(
    fn: Callable[P, R],
    *,
    defaults: tuple[str, ...],
) -> Callable[P, R]:
    sig = signature(fn)
    try:
        variant_index = [
            idx
            for idx, parameter in enumerate(sig.parameters.values())
            if parameter.name == "variants"
        ][0]
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Function is missing a 'variants' parameter") from exc

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args_list = list(args)
        if len(args_list) > variant_index:
            if args_list[variant_index] is None:
                args_list[variant_index] = list(defaults)
        else:
            if kwargs.get("variants") is None:
                kwargs["variants"] = list(defaults)
        return fn(*args_list, **kwargs)

    return wrapper


def build_cvc_preset(default_variants: Sequence[str]) -> CvcPreset:
    """Create a wrapper set that pins default variants for CoGs vs Clips recipes."""

    defaults = tuple(default_variants)

    base_functions: dict[str, Callable[..., object]] = {
        "make_eval_suite": base_make_eval_suite,
        "make_training_env": base_make_training_env,
        "make_curriculum": base_make_curriculum,
        "train": base_train,
        "train_small_maps": base_train_small_maps,
        "train_medium_maps": base_train_medium_maps,
        "train_large_maps": base_train_large_maps,
        "train_coordination": base_train_coordination,
        "train_single_mission": base_train_single_mission,
        "evaluate": base_evaluate,
        "play": base_play,
        "play_training_env": base_play_training_env,
    }

    wrapped_functions = {
        name: _wrap_with_default_variants(func, defaults=defaults)
        for name, func in base_functions.items()
    }

    return CvcPreset(
        default_variants=defaults,
        **wrapped_functions,  # type: ignore[arg-type]
    )


__all__ = ["CvcPreset", "build_cvc_preset"]
