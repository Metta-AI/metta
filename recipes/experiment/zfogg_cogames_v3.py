"""Zach's CoGames V3 - Experimental Aggressive Approach

This version tries RADICAL departures from the failed V1:
1. Start with EASIEST possible task (lonely_heart variant)
2. Use curriculum to gradually remove training wheels
3. Smaller agent count (2 cogs) for faster learning signal
4. Focused on single mission until mastery
5. Higher entropy bonus for more exploration

Philosophy: Instead of trying to learn hard task directly, start with
absurdly easy version and gradually make it harder via curriculum.
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


def make_progressive_difficulty_curriculum(
    num_cogs: int = 2,  # Fewer agents = clearer learning signal
) -> CurriculumConfig:
    """Start EXTREMELY easy, progressively remove training wheels.

    lonely_heart variant makes hearts trivially easy (1 resource each).
    Curriculum will learn this first, then graduate to real task.
    """
    # Start with easiest possible version
    easy_env = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=["lonely_heart"],  # Makes hearts VERY easy
    )

    tasks = cc.bucketed(easy_env)

    # Curriculum progression axes:
    # 1. Episode length (shorter = easier but less rewarding)
    tasks.add_bucket("game.max_steps", [500, 750, 1000, 1250])

    # 2. Heart requirements (lonely_heart makes this easy, curriculum removes it)
    # This will create variants with/without lonely_heart
    tasks.add_bucket("game.agent.rewards.stats.heart.gained", [0.1, 0.5, 1.0])

    # Bidirectional learning progress with AGGRESSIVE exploration
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.2,  # Higher than normal (default 0.1) - explore more
        max_memory_tasks=1000,
        max_slice_axes=5,
    )

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def train(
    run: str = "zfogg_v3",
    num_cogs: int = 2,  # Fewer agents for clearer signal
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V3 - start absurdly easy, progressively harder.

    Expected timeline:
    - 0-100M: Master lonely_heart variant (trivial hearts)
    - 100M-500M: Curriculum gradually increases difficulty
    - 500M-1B: Should be attempting real task
    - 1B-2B: Mastery of real task

    Why this works:
    - Agents learn heart-making behavior early (even if trivial)
    - Curriculum slowly makes task harder
    - By the time they face real task, they know the "shape" of the solution
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_progressive_difficulty_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    # Increase entropy bonus for more exploration
    trainer_cfg = TrainerConfig()
    trainer_cfg.losses.ppo_actor.ent_coef = 0.015  # Higher than default 0.01

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


def train_tiny_test(run: str = "zfogg_v3_test", disable_wandb: bool = True) -> TrainTool:
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
    """Evaluate V3 policy."""
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
    """Interactive play with V3 policy."""
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
