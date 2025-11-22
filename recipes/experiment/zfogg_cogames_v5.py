"""Zach's CoGames V5 - Proven Techniques from Successful Recipes

Why V3/V4 failed (both 0.00 hearts after 400M-800M steps):
- V3: Reward coefficient bucketing diluted learning signal
- V4: lonely_heart is TOO different from real task, no transfer

V5 uses PROVEN techniques from working recipes:
1. heart_chorus: Proven reward shaping (hearts + inventory diversity)
2. pack_rat: Remove capacity constraints (focus on strategy, not inventory mgmt)
3. tiny_heart_protocols: Cheaper heart recipes (easier initial success)
4. inventory_heart_tune: Start with resources (bootstrap learning)
5. compass: Help agents find assembler
6. small_50: Smaller maps initially (less navigation)

Philosophy: Make initial success EASY, then gradually remove training wheels.
Train on REAL task (not fake easy variant), just with helpful scaffolding.
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


def make_scaffolded_curriculum(
    num_cogs: int = 2,
) -> CurriculumConfig:
    """Curriculum with training wheels that teach the REAL task.

    Unlike lonely_heart (which changes the task), these variants make
    the SAME task easier by:
    - Providing starting resources (inventory_heart_tune)
    - Cheaper recipes (tiny_heart_protocols)
    - Unlimited capacity (pack_rat)
    - Smaller maps (small_50)
    - Navigation help (compass)
    - Reward shaping (heart_chorus)

    The policy learns the REAL behavior (collect → craft → deposit)
    but with helpful scaffolding.
    """
    # Start with FULL training wheels
    env = make_training_env(
        num_cogs=num_cogs,
        mission="extractor_hub_30",  # Easiest mission
        variants=[
            "heart_chorus",  # Reward shaping: hearts + inventory diversity
            "pack_rat",  # Max capacity (255) - no inventory mgmt
            "tiny_heart_protocols",  # Cheaper heart recipes
            "inventory_heart_tune",  # Start with resources (default: 1 heart worth)
            "compass",  # Help find assembler
            "small_50",  # 50x50 map (less navigation)
        ],
    )

    tasks = cc.bucketed(env)

    # Curriculum axis: episode length (controls difficulty without changing task)
    # Shorter episodes = less time to make hearts = need to be more efficient
    tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

    # Standard learning progress algorithm (proven in cvc_arena)
    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.1,  # Default, not inflated
        max_memory_tasks=1000,
        max_slice_axes=5,
    )

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def train(
    run: str = "zfogg_v5",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Train V5 - scaffolded learning on the REAL task.

    Expected timeline:
    - 0-100M: Should start seeing hearts (thanks to training wheels)
    - 100M-500M: Consistent hearts on scaffolded tasks
    - 500M-1B: Transfer to standard difficulty (no training wheels)

    Key differences from V4:
    - NO lonely_heart (it doesn't transfer)
    - Uses proven reward shaping (heart_chorus)
    - Trains on REAL task with helpful scaffolding
    - Smaller maps, starting resources, cheaper recipes
    - Same fundamental behavior as final task
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_scaffolded_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    # Default trainer config (proven to work)
    trainer_cfg = TrainerConfig()

    # Evaluate on STANDARD difficulty (ultimate goal)
    # But also evaluate on training tasks to see learning progress
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=None,  # No training wheels in eval
    )

    evaluator_cfg = EvaluatorConfig(simulations=eval_suite)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=policy_architecture,
    )


def train_with_eval_variants(
    run: str = "zfogg_v5_staged",
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
    disable_wandb: bool = False,
) -> TrainTool:
    """Alternative: Also evaluate on scaffolded tasks to track progress.

    This helps you see if the policy is learning ANYTHING, even if it
    can't transfer to standard difficulty yet.
    """
    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    curriculum = make_scaffolded_curriculum(num_cogs=num_cogs)

    if disable_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    trainer_cfg = TrainerConfig()

    # Evaluate on BOTH scaffolded and standard difficulty
    eval_suite_scaffolded = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=["heart_chorus", "pack_rat", "tiny_heart_protocols", "small_50"],
    )

    eval_suite_standard = make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        variants=None,
    )

    # Combine both eval suites
    from metta.sim.simulation_config import SimulationSuiteConfig

    combined_suite = SimulationSuiteConfig(
        name="v5_combined_eval",
        simulations=[
            *eval_suite_scaffolded.simulations,
            *eval_suite_standard.simulations,
        ],
    )

    evaluator_cfg = EvaluatorConfig(simulations=combined_suite)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        policy_architecture=policy_architecture,
    )


def train_tiny_test(run: str = "zfogg_v5_test", disable_wandb: bool = True) -> TrainTool:
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
    """Evaluate V5 policy."""
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
    with_scaffolding: bool = True,
) -> PlayTool:
    """Interactive play with V5 policy.

    Args:
        with_scaffolding: If True, use training variants. If False, use standard difficulty.
    """
    from metta.sim.simulation_config import SimulationConfig

    variants = None
    if with_scaffolding:
        variants = [
            "heart_chorus",
            "pack_rat",
            "tiny_heart_protocols",
            "inventory_heart_tune",
            "compass",
            "small_50",
        ]

    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission}_{num_cogs}cogs",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


__all__ = [
    "train",
    "train_with_eval_variants",
    "train_tiny_test",
    "evaluate",
    "play",
]
