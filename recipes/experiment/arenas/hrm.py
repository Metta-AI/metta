"""Arena recipe with HRM policy architecture."""

from metta.agent.policy import PolicyArchitecture
from recipes.experiment import arena as base
from recipes.experiment.architectures import get_architecture

mettagrid = base.mettagrid
make_curriculum = base.make_curriculum
simulations = base.simulations
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
    tool = base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )
    tool = tool.model_copy(update={"policy_architecture": policy_architecture or get_architecture("hrm_tiny")})
    return tool


def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    tool = base.train_shaped(rewards=rewards, converters=converters)
    tool = tool.model_copy(update={"policy_architecture": policy_architecture or get_architecture("hrm_tiny")})
    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "train",
    "train_shaped",
]
