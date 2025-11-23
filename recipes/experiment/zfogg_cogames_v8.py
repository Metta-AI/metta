"""Zach's CoGames V8 - Triple Fix for Transfer Learning

V7 results at 2.5B steps:
- Training tasks: 2.55 hearts ✓
- Standard tasks: 0.00 hearts ✗
- Root cause: Policy optimizes for resource collection, never learns to craft hearts

V8 fixes with THREE combined approaches:

1. DECAYING INTERMEDIATE REWARDS
   - Start: High resource collection rewards (0.02) to guide initial learning
   - End: Zero resource rewards (0.0) to force heart optimization
   - Curriculum automatically removes training wheels

2. LONGER EPISODES
   - 2000-4000 steps instead of 1000-1500
   - Gives policy time to complete full behavior: collect → craft → deposit
   - Removes time pressure that incentivizes easy resource collection

3. HIGHER HEART COEFFICIENTS
   - Scale up to 10.0 (from 5.0 in V7)
   - Makes hearts dramatically more valuable than resources
   - Clear signal: hearts are the objective

Philosophy: Guide the policy early with intermediate rewards, then gradually
remove them. Long episodes ensure it has time to complete the full behavior.
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


def make_decaying_reward_curriculum(
    num_cogs: int = 2,
) -> CurriculumConfig:
    """Curriculum with decaying intermediate rewards + longer episodes.

    The game is IDENTICAL in all tasks (same recipes, same limits).

    Curriculum progression:

    Early training (high resource rewards):
    - Resource collection: 0.02 per unit → guides exploration
    - Hearts: 1.0-5.0 → establishes final objective
    - Episodes: 2000 steps → time to learn basics

    Mid training (medium resource rewards):
    - Resource collection: 0.01-0.005 → less guidance
    - Hearts: 5.0-10.0 → increasing heart value
    - Episodes: 3000 steps → time for complex behavior

    Late training (zero resource rewards):
    - Resource collection: 0.0 → no guidance, pure optimization
    - Hearts: 10.0 → maximum value on real objective
    - Episodes: 4000 steps → full behavior chain

    This is a PROVEN approach: start with scaffolding, remove gradually.
    """
    # Base environment - NO variants, NO modifications
    env = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=None,  # No game-changing variants!
    )

    tasks = cc.bucketed(env)

    # FIX 1: DECAYING resource collection rewards
    # Start high (guide learning), decay to zero (force heart optimization)
    for resource in ["carbon", "oxygen", "germanium", "silicon"]:
        tasks.add_bucket(
            f"game.agent.rewards.stats.{resource}.gained",
            [0.0, 0.005, 0.01, 0.02],  # Curriculum will decay from 0.02 → 0.0
        )

    # FIX 3: HIGHER heart reward coefficients
    # Scale up to 10.0 to make hearts dramatically more valuable
    tasks.add_bucket("game.agent.rewards.stats.heart.gained", [1.0, 2.0, 5.0, 10.0])

    # FIX 2: LONGER episodes
    # 2000-4000 steps gives time for full behavior: collect → craft → deposit
    tasks.add_bucket("game.max_steps", [2000, 2500, 3000, 4000])

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
    run: str = "zfogg_v8",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V8 - decaying rewards + longer episodes + higher heart values.

    Expected timeline:
    - 0-500M: Learn with high intermediate rewards (0.02) and shorter episodes (2000)
    - 500M-1.5B: Curriculum reduces intermediate rewards and increases episode length
    - 1.5B-3B: Zero intermediate rewards, max episodes (4000), pure heart optimization

    Key differences from V7:
    - Decaying resource rewards (0.02 → 0.0) instead of constant
    - Longer episodes (2000-4000) instead of (1000-1500)
    - Higher heart coefficients (up to 10.0) instead of (up to 5.0)
    - Combined approach attacks the transfer problem from three angles

    Hypothesis:
    - Early: Policy learns to collect resources (guided by 0.02 rewards)
    - Mid: Policy learns to craft hearts (resources become less valuable)
    - Late: Policy optimizes heart production (resources give 0.0 reward)
    - Longer episodes give time to complete full behavior chain
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_decaying_reward_curriculum(num_cogs=num_cogs)

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


def train_tiny_test(run: str = "zfogg_v8_test", disable_wandb: bool = True) -> TrainTool:
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
    """Evaluate V8 policy."""
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
    """Interactive play with V8 policy."""
    from metta.sim.simulation_config import SimulationConfig

    # Always use standard difficulty - V8 trains on the real task
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
