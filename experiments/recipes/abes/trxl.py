from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep,
    train as base_train,
)
from metta.agent.policies.trxl import TRXLConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import OptimizerConfig

DEFAULT_LR = OptimizerConfig.model_fields["learning_rate"].default


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    if policy_architecture is None:
        policy_architecture = TRXLConfig()

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    # TRXLConfig is a direct PolicyArchitecture; keep optimizer defaults.

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
