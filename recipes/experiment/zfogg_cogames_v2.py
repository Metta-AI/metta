"""Zach's CoGames V2 - Actually Working Policy

Based on analysis of failed V1 approach, this recipe:
1. NO reward shaping - forces learning the real task
2. Uses bidirectional learning progress curriculum (proven effective)
3. Simpler mission set focused on core skills
4. Better architecture defaults
5. More aggressive exploration

The key insight: Reward shaping was making the policy learn to game
inventory diversity instead of actually making hearts.
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

# CRITICAL: NO reward shaping variants
# The policy MUST learn to actually make hearts, not game diversity rewards
CORE_MISSIONS: list[str] = [
    # Start with simplest extraction tasks
    "extractor_hub_30",
    "extractor_hub_50",
    # Add resource collection variety
    "collect_resources_classic",
    "collect_resources_spread",
    # Introduce constraints
    "oxygen_bottleneck",
    "energy_starved",
]


def make_better_curriculum(
    num_cogs: int = 4,
    missions: Optional[list[str]] = None,
) -> CurriculumConfig:
    """Create curriculum using bidirectional learning progress.

    This is the same algorithm used in cvc_arena which actually works.
    Key differences from failed approach:
    - NO reward shaping variants
    - Bidirectional learning progress (explores both easy and hard tasks)
    - Bucketed curriculum with max_steps progression
    - Higher exploration bonus
    """
    if missions is None:
        missions = list(CORE_MISSIONS)

    all_mission_tasks = []
    for mission_name in missions:
        # NO VARIANTS = no reward shaping = learn the real task
        mission_env = make_training_env(
            num_cogs=num_cogs,
            mission=mission_name,
            variants=None,  # CRITICAL: No reward shaping!
        )

        mission_tasks = cc.bucketed(mission_env)

        # Add curriculum buckets for episode length
        # Shorter episodes are easier but less rewarding
        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

        all_mission_tasks.append(mission_tasks)

    # Merge all missions into single curriculum
    merged_tasks = cc.merge(all_mission_tasks)

    # Use bidirectional learning progress (proven effective in cvc_arena)
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,  # Explore both easy and hard tasks
        ema_timescale=0.001,  # Smooth learning progress estimates
        exploration_bonus=0.15,  # Higher than default 0.1 for more exploration
        max_memory_tasks=1000,
        max_slice_axes=5,
        enable_detailed_slice_logging=False,
    )

    return merged_tasks.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    run: str = "zfogg_v2",
    num_cogs: int = 4,
    missions: Optional[list[str]] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V2 policy - no reward shaping, better curriculum.

    Expected behavior:
    - First 500M steps: Policy will struggle, get 0.00 hearts (this is GOOD)
    - 500M-1B steps: Should start seeing occasional hearts
    - 1B-2B steps: Should get consistent hearts on easier tasks
    - 2B+ steps: Should generalize to harder tasks

    The V1 policy was getting 0.50-0.61 from reward shaping but 0.00 real hearts.
    V2 will look worse initially but actually learn the task.
    """
    if missions is None:
        missions = list(CORE_MISSIONS)

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_better_curriculum(
        num_cogs=num_cogs,
        missions=missions,
    )

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    # Use default trainer config (already well-tuned)
    trainer_cfg = TrainerConfig()

    # Evaluate without any reward shaping
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=None,  # No reward shaping in eval
    )

    evaluator_cfg = EvaluatorConfig(simulations=eval_suite)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=policy_architecture,
    )


def train_tiny_test(run: str = "zfogg_v2_test", disable_wandb: bool = True) -> TrainTool:
    """Tiny test - just check it runs."""
    return train(
        run=run,
        num_cogs=2,
        missions=["extractor_hub_30"],
        disable_wandb=disable_wandb,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
) -> EvaluateTool:
    """Evaluate V2 policy."""
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
    """Interactive play with V2 policy."""
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
