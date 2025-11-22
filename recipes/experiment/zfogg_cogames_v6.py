"""Zach's CoGames V6 - Gradual Scaffolding Removal

V5 results at 824M steps:
- Training tasks (with scaffolding): 2.42 hearts ✓
- Standard tasks (no scaffolding): 0.00 hearts ✗

V5 proved the scaffolding works! The policy CAN learn to make hearts.
But it's not transferring to standard difficulty yet.

V6 solution: GRADUALLY remove scaffolding via curriculum.
Start with full training wheels, slowly remove them as policy improves.

Scaffolding removal order (easiest → hardest):
1. Remove small_50 (use full-size maps)
2. Remove compass (no navigation help)
3. Remove inventory_heart_tune (no starting resources)
4. Remove tiny_heart_protocols (use normal heart costs)
5. Remove pack_rat (normal capacity limits)
6. Remove heart_chorus (no reward shaping) - FINAL difficulty

This way the policy learns the SAME task, just with gradually
increasing difficulty.
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


def make_gradual_curriculum(
    num_cogs: int = 2,
) -> CurriculumConfig:
    """Curriculum that gradually removes scaffolding.

    Creates tasks with different levels of scaffolding:
    Level 0: Full scaffolding (all 6 variants)
    Level 1: Remove small_50 + compass (navigation difficulty)
    Level 2: Remove inventory_heart_tune (no starting resources)
    Level 3: Remove tiny_heart_protocols (normal heart costs)
    Level 4: Remove pack_rat (normal capacity)
    Level 5: Remove heart_chorus (no reward shaping) = STANDARD

    Learning progress algorithm will automatically select easier tasks
    when policy struggles and harder tasks when it succeeds.
    """
    # Level 0: Full scaffolding (easiest)
    env_level0 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=[
            "heart_chorus",
            "pack_rat",
            "tiny_heart_protocols",
            "inventory_heart_tune",
            "compass",
            "small_50",
        ],
    )

    # Level 1: Remove navigation helpers
    env_level1 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=[
            "heart_chorus",
            "pack_rat",
            "tiny_heart_protocols",
            "inventory_heart_tune",
        ],
    )

    # Level 2: Remove starting resources
    env_level2 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=[
            "heart_chorus",
            "pack_rat",
            "tiny_heart_protocols",
        ],
    )

    # Level 3: Remove cheap recipes
    env_level3 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=[
            "heart_chorus",
            "pack_rat",
        ],
    )

    # Level 4: Remove unlimited capacity
    env_level4 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=[
            "heart_chorus",
        ],
    )

    # Level 5: No scaffolding (standard difficulty)
    env_level5 = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",
        variants=None,
    )

    # Merge all levels into a single curriculum
    all_envs = [env_level0, env_level1, env_level2, env_level3, env_level4, env_level5]
    all_tasks = [cc.bucketed(env) for env in all_envs]

    # Add episode length variation to each level
    for tasks in all_tasks:
        tasks.add_bucket("game.max_steps", [1000, 1250, 1500])

    merged = cc.merge(all_tasks)

    # Use bidirectional learning progress to automatically select difficulty
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.1,
        max_memory_tasks=1000,
        max_slice_axes=5,
    )

    return merged.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    run: str = "zfogg_v6",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V6 - gradual scaffolding removal.

    Expected timeline:
    - 0-200M: Master level 0-1 (full scaffolding, then no nav help)
    - 200M-500M: Progress through levels 2-3 (normal resources/recipes)
    - 500M-1B: Master level 4-5 (normal capacity, then pure standard)

    The curriculum automatically adjusts difficulty based on policy performance.
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_gradual_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    trainer_cfg = TrainerConfig()

    # Evaluate on standard difficulty (the goal)
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


def train_tiny_test(run: str = "zfogg_v6_test", disable_wandb: bool = True) -> TrainTool:
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
    """Evaluate V6 policy."""
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
    scaffolding_level: int = 0,
) -> PlayTool:
    """Interactive play with V6 policy.

    Args:
        scaffolding_level: 0 (full) to 5 (none/standard)
    """
    from metta.sim.simulation_config import SimulationConfig

    variants_by_level = {
        0: ["heart_chorus", "pack_rat", "tiny_heart_protocols", "inventory_heart_tune", "compass", "small_50"],
        1: ["heart_chorus", "pack_rat", "tiny_heart_protocols", "inventory_heart_tune"],
        2: ["heart_chorus", "pack_rat", "tiny_heart_protocols"],
        3: ["heart_chorus", "pack_rat"],
        4: ["heart_chorus"],
        5: None,
    }

    variants = variants_by_level.get(scaffolding_level, None)

    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission}_{num_cogs}cogs_level{scaffolding_level}",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


__all__ = [
    "train",
    "train_tiny_test",
    "evaluate",
    "play",
]
