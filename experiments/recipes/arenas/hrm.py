"""Arena recipe with HRM policy architecture."""

from metta.agent.policies.hrm import HRMTinyConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import OptimizerConfig
from experiments.recipes import arena as base

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
    # Update policy architecture
    tool = tool.model_copy(
        update={"policy_architecture": policy_architecture or HRMTinyConfig()}
    )
    return tool


def train_shaped(
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
):
    """Train with HRM policy architecture using shaped rewards (defaults to HRMTinyConfig)."""
    tool = base.train_shaped(rewards=rewards, converters=converters)

    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,  # Small weight decay for AdamW
        warmup_steps=2000,  # Warmup steps for ScheduleFree
    )

    # Update trainer config with new optimizer
    trainer_config = tool.trainer.model_copy(update={"optimizer": optimizer_config})

    # Update policy architecture and trainer
    tool = tool.model_copy(
        update={
            "policy_architecture": policy_architecture or HRMTinyConfig(),
            "trainer": trainer_config,
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
