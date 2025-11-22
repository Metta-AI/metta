"""Zach's CoGames V4 - Simplified Curriculum Fix

V3 failed because:
1. Reward coefficient bucket diluted learning signal
2. Too much exploration (high entropy + exploration bonus)
3. Curriculum complexity prevented learning the basic task

V4 fixes:
1. Remove reward coefficient bucketing - use full reward signal
2. Reduce exploration back to reasonable levels
3. Simpler curriculum - just episode length progression
4. Keep lonely_heart variant throughout training
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import make_eval_suite, make_training_env


def make_simple_curriculum(
    num_cogs: int = 2,
) -> CurriculumConfig:
    """Ultra-simple curriculum: lonely_heart variant + episode length only.

    The lonely_heart variant makes hearts trivially easy (1 resource each).
    We just vary episode length to control difficulty.
    NO reward shaping, NO coefficient buckets - keep the learning signal clear.
    """
    # Start with easiest possible version
    easy_env = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=["lonely_heart"],  # Makes hearts VERY easy
    )

    tasks = cc.bucketed(easy_env)

    # ONLY vary episode length - this is the simplest curriculum axis
    # Shorter episodes are easier but less rewarding
    tasks.add_bucket("game.max_steps", [500, 750, 1000, 1250, 1500])

    # Use learning progress but with NORMAL exploration (not aggressive)
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.1,  # Default, not inflated
        max_memory_tasks=1000,
        max_slice_axes=5,
    )

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def train(
    run: str = "zfogg_v4",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V4 - simplified curriculum, normal exploration.

    Expected behavior:
    - 0-100M steps: Should start seeing hearts on lonely_heart variant
    - 100M-500M: Should get consistent hearts on easy tasks
    - 500M+: May need to graduate to harder variants

    Changes from V3:
    - Removed reward coefficient bucketing (was diluting signal)
    - Reduced exploration back to normal levels
    - Simpler curriculum with just episode length variation
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_simple_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    # Use DEFAULT entropy coefficient (not inflated)
    trainer_cfg = TrainerConfig()
    # trainer_cfg.losses.ppo_actor.ent_coef stays at default 0.01

    # Evaluate on STANDARD difficulty (no training wheels)
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=None,
    )

    evaluator_cfg = EvaluatorConfig(simulations=eval_suite)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=policy_architecture,
    )


def train_tiny_test(run: str = "zfogg_v4_test", disable_wandb: bool = True) -> TrainTool:
    """Quick test."""
    return train(
        run=run,
        num_cogs=2,
        disable_wandb=disable_wandb,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 2,
) -> EvaluateTool:
    """Evaluate V4 policy."""
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
    mission: str = "extractor_hub_30",
    num_cogs: int = 2,
) -> PlayTool:
    """Interactive play with V4 policy."""
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
    "train_tiny_test",
    "evaluate",
    "play",
]
