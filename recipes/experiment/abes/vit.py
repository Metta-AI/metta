from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
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
    curriculum=None,
    policy_architecture: PolicyArchitecture | None = None,
):
    return base_train(
        curriculum=curriculum,
        policy_architecture=policy_architecture or ViTDefaultConfig(),
    )


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
