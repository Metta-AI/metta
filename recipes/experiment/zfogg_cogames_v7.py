"""Zach's CoGames V7 - Reward Curriculum (Proven Approach)

Why V5/V6 failed (both stuck at 0.00 on standard after 800M-2.5B steps):
- Used variants that CHANGE THE GAME (tiny_heart_protocols, pack_rat, lonely_heart)
- Policy learned a different task that doesn't transfer
- Scaffolding was too different from final objective

V7 uses the PROVEN approach from arena_basic_easy_shaped:
- Keep game mechanics IDENTICAL (same recipes, same limits)
- Vary REWARD COEFFICIENTS to guide learning
- Curriculum on reward weights, not task variants

Philosophy: The policy learns the EXACT SAME TASK from the start.
We just give it more feedback (intermediate rewards) initially,
then gradually remove that extra feedback.
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


def make_reward_curriculum(
    num_cogs: int = 2,
) -> CurriculumConfig:
    """Curriculum that varies REWARD COEFFICIENTS, not game mechanics.

    The game is IDENTICAL in all tasks (same recipes, same limits).
    We just provide different amounts of intermediate feedback:

    High coefficients (early learning):
    - Reward collecting each resource type
    - Reward inventory diversity
    - Reward approaching assembler
    - Guide the policy toward the right behavior

    Low coefficients (later learning):
    - Remove intermediate rewards
    - Only reward final objective (hearts)
    - Force policy to optimize for the real goal

    This is the PROVEN approach from arena_basic_easy_shaped.
    """
    # Base environment - NO variants, NO modifications
    # This is the REAL task the policy needs to solve
    env = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=None,  # No game-changing variants!
    )

    tasks = cc.bucketed(env)

    # Curriculum Axis 1: Reward coefficients for resource collection
    # These are PROVEN to work from cvc/mission_variant_curriculum.py
    #
    # High values = lots of guidance for collecting resources
    # Low values = only reward the final objective (hearts)

    # Resource collection rewards (encourages collecting each type)
    # Start with higher coefficients, curriculum reduces them over time
    for resource in ["carbon", "oxygen", "germanium", "silicon"]:
        tasks.add_bucket(f"game.agent.rewards.stats.{resource}.gained", [0.0, 0.005, 0.01, 0.02])

    # Heart reward coefficient (the final objective)
    # This should ALWAYS be high - it's the real goal
    tasks.add_bucket("game.agent.rewards.stats.heart.gained", [1.0, 2.0, 5.0])

    # Curriculum Axis 2: Episode length (controls sample efficiency)
    tasks.add_bucket("game.max_steps", [1000, 1250, 1500])

    # Use bidirectional learning progress (proven in production recipes)
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.1,
        max_memory_tasks=1000,
        max_slice_axes=5,
    )

    return tasks.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    run: str = "zfogg_v7",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V7 - reward curriculum on the REAL task.

    Expected timeline:
    - 0-500M: Learn with high intermediate rewards (collect resources, diversity)
    - 500M-1B: Curriculum reduces intermediate rewards, focuses on hearts
    - 1B-2B: Should master the task with minimal guidance

    Key differences from V5/V6:
    - NO game-changing variants (tiny_heart_protocols, pack_rat, lonely_heart)
    - Same game mechanics from the start (same recipes, same limits)
    - Only varies reward coefficients to provide guidance
    - Follows proven pattern from arena_basic_easy_shaped
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_reward_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    trainer_cfg = TrainerConfig()

    # Evaluate on standard difficulty (exactly what we're training on!)
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


def train_tiny_test(run: str = "zfogg_v7_test", disable_wandb: bool = True) -> TrainTool:
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
    """Evaluate V7 policy."""
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
    """Interactive play with V7 policy."""
    from metta.sim.simulation_config import SimulationConfig

    # Always use standard difficulty - V7 trains on the real task
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
