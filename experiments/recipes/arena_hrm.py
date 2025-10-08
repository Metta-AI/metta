"""Arena recipe with HRM policy architecture."""

from metta.agent.policies.hrm import HRMTinyConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.loss.hrm_ppo import HRMPPOConfig

from experiments.recipes import arena as base

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
    tool = base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )
    # Update policy architecture
    tool = tool.model_copy(
        update={"policy_architecture": policy_architecture or HRMTinyConfig()}
    )
    # Update loss config to use HRMPPO
    tool.trainer.losses.loss_configs = {"hrm_ppo": HRMPPOConfig()}
    return tool


def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    tool = base.train_shaped(rewards=rewards, converters=converters)
    # Update policy architecture
    tool = tool.model_copy(
        update={"policy_architecture": policy_architecture or HRMTinyConfig()}
    )
    # Update loss config to use HRMPPO
    tool.trainer.losses.loss_configs = {"hrm_ppo": HRMPPOConfig()}
    return tool


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
