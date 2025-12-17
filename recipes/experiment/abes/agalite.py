"""Arena Basic Easy Shaped recipe targeting AGaLiTe policy variants."""

from __future__ import annotations

from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import OptimizerConfig
from metta.tools.train import TrainTool
from recipes.experiment.architectures import get_architecture
from recipes.prod.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep,
)
from recipes.prod.arena_basic_easy_shaped import (
    train as base_train,
)


def train(
    *,
    curriculum: CurriculumConfig | None = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    agent: str | None = None,
) -> TrainTool:
    if policy_architecture is None:
        if agent is not None and agent != "agalite":
            raise ValueError(f"Unknown agent '{agent}'. Available: agalite")
        policy_architecture = get_architecture("agalite")

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
    "sweep",
    "train",
]
