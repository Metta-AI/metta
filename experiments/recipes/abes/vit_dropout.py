from metta.agent.policies.vit_dropout import ViTDropoutConfig
from metta.agent.policy import PolicyArchitecture

from experiments.recipes import arena as base
from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep,
)
from experiments.recipes.arena_basic_easy_shaped import (
    train as base_train,
)


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    return base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture or ViTDropoutConfig(),
    )



def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    tool = base.train_shaped(rewards=rewards, converters=converters)
    # Update policy architecture
    tool = tool.model_copy(
        update={"policy_architecture": policy_architecture or ViTDropoutConfig()}
    )
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
    "train_shaped",
]
