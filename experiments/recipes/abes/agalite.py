"""Arena Basic Easy Shaped recipe targeting AGaLiTe policy variants."""

from __future__ import annotations

from typing import Callable

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
    train as base_train,
)
from metta.agent.policies.agalite import AGaLiTeConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import OptimizerConfig
from metta.tools.train import TrainTool

_POLICY_PRESETS: dict[str, Callable[[], PolicyArchitecture]] = {
    "agalite": AGaLiTeConfig,
}


def _policy_from_name(name: str) -> PolicyArchitecture:
    try:
        return _POLICY_PRESETS[name]()
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(_POLICY_PRESETS))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}") from exc


def train(
    *,
    curriculum: CurriculumConfig | None = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    agent: str | None = None,
) -> TrainTool:
    if policy_architecture is None:
        if agent is not None:
            policy_architecture = _policy_from_name(agent)
        else:
            policy_architecture = AGaLiTeConfig()

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    hint = getattr(policy_architecture, "learning_rate_hint", None)
    optimizer = tool.trainer.optimizer
    default_lr = OptimizerConfig.model_fields["learning_rate"].default
    if hint is not None and optimizer.learning_rate == default_lr:
        optimizer.learning_rate = hint

    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
