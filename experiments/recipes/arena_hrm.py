"""Arena recipe with HRM policy architecture."""

from experiments.recipes import arena as base
from metta.agent.policies.hrm import HRMTinyConfig
from metta.agent.policy import PolicyArchitecture

make_mettagrid = base.make_mettagrid
make_curriculum = base.make_curriculum
make_evals = base.make_evals
play = base.play
replay = base.replay
evaluate = base.evaluate


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture (defaults to HRMTinyConfig for memory efficiency)."""
    return base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    ).model_copy(update={"policy_architecture": policy_architecture or HRMTinyConfig()})


def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    return base.train_shaped(rewards=rewards, converters=converters).model_copy(
        update={"policy_architecture": policy_architecture or HRMTinyConfig()}
    )


__all__ = [
    "make_mettagrid",
    "make_curriculum",
    "make_evals",
    "play",
    "replay",
    "evaluate",
    "train",
    "train_shaped",
]
