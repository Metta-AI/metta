"""Zach's experimental CoGames policy training recipe.

This is a personal scratchpad for experimenting with CoGames policies.
Based on the simple_cogs_policy recipe with room for customization.
"""

from __future__ import annotations

from typing import Optional, Sequence

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import (
    make_curriculum,
    make_eval_suite,
    make_training_env,
)
from recipes.experiment.simple_cogs_policy import (
    EVAL_VARIANTS,
    SIMPLE_COGS_MISSIONS,
    TRAINING_VARIANTS,
)


def train(
    run: str = "zfogg_cogames_v1",
    num_cogs: int = 4,
    missions: Optional[list[str]] = None,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """My experimental cogames training.

    Customize this function to try different:
    - Mission combinations
    - Number of agents
    - Variants for reward shaping
    - Policy architectures
    """
    if missions is None:
        missions = list(SIMPLE_COGS_MISSIONS)

    if variants is None:
        variants = list(TRAINING_VARIANTS)

    if eval_variants is None:
        eval_variants = list(EVAL_VARIANTS)

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_curriculum(
        num_cogs=num_cogs,
        missions=missions,
        variants=variants,
    )

    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig

    # Configure W&B - disable for scratchpad experiments if desired
    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    trainer_cfg = TrainerConfig()

    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=eval_variants,
    )

    evaluator_cfg = EvaluatorConfig(simulations=eval_suite)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=policy_architecture,
    )


def train_quick_test(run: str = "zfogg_quick_test", disable_wandb: bool = True) -> TrainTool:
    """Quick training run for testing.

    By default disables W&B for quick tests - set disable_wandb=False to enable.
    """
    return train(
        run=run,
        num_cogs=2,
        missions=["extractor_hub_30"],
        variants=["heart_chorus", "pack_rat"],
        disable_wandb=disable_wandb,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
) -> EvaluateTool:
    """Evaluate my trained policy."""
    return EvaluateTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty="standard",
            variants=None,
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "extractor_hub_50",
    num_cogs: int = 4,
) -> PlayTool:
    """Interactive play with my trained policy."""
    from metta.sim.simulation_config import SimulationConfig

    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=None,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission}_{num_cogs}cogs",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


__all__ = [
    "train",
    "train_quick_test",
    "evaluate",
    "play",
]
