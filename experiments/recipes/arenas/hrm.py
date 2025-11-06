"""Arena recipe with HRM policy architecture."""

import metta.agent.policies.hrm
import metta.agent.policy

import experiments.recipes

mettagrid = experiments.recipes.arena.mettagrid
make_curriculum = experiments.recipes.arena.make_curriculum
simulations = experiments.recipes.arena.simulations
play = experiments.recipes.arena.play
replay = experiments.recipes.arena.replay
evaluate = experiments.recipes.arena.evaluate


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture (defaults to HRMTinyConfig for memory efficiency)."""
    tool = experiments.recipes.arena.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )
    # Update policy architecture
    tool = tool.model_copy(
        update={
            "policy_architecture": policy_architecture
            or metta.agent.policies.hrm.HRMTinyConfig()
        }
    )
    return tool


def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    tool = experiments.recipes.arena.train_shaped(
        rewards=rewards, converters=converters
    )
    # Update policy architecture
    tool = tool.model_copy(
        update={
            "policy_architecture": policy_architecture
            or metta.agent.policies.hrm.HRMTinyConfig()
        }
    )
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
