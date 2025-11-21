"""Simple but effective CoGames policy for Cogs vs Clips.

This recipe creates a policy trained on resource gathering and coordination missions
with reward shaping to accelerate learning. It uses a progressive curriculum starting
from easier tasks and building up to more complex coordination challenges.
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

# Progressive mission curriculum: start simple, increase complexity
SIMPLE_COGS_MISSIONS: list[str] = [
    # Start with small extraction hubs to learn basic resource gathering
    "extractor_hub_30",
    "extractor_hub_50",
    # Progress to resource collection with more variety
    "collect_resources_classic",
    "collect_resources_spread",
    # Add constraint handling
    "oxygen_bottleneck",
    "energy_starved",
    # Build up to coordination challenges
    "divide_and_conquer",
    "collect_far",
]

# Reward shaping variants to accelerate learning
TRAINING_VARIANTS: list[str] = [
    "heart_chorus",  # Reward shaping for heart production and inventory diversity
    "pack_rat",  # Increased capacity to reduce resource management difficulty
]

# Evaluation without training aids to measure true performance
EVAL_VARIANTS: list[str] = []


def train(
    num_cogs: int = 4,
    missions: Optional[list[str]] = None,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train a simple but effective CoGames policy.

    Args:
        num_cogs: Number of agents per mission (default: 4 for good cooperation dynamics)
        missions: Mission list for curriculum (defaults to SIMPLE_COGS_MISSIONS)
        variants: Training variants for reward shaping (defaults to TRAINING_VARIANTS)
        eval_variants: Evaluation variants (defaults to no shaping)
        policy_architecture: Neural architecture (defaults to ViTDefaultConfig)
        enable_detailed_slice_logging: Enable detailed curriculum logging
    """
    if missions is None:
        missions = list(SIMPLE_COGS_MISSIONS)

    if variants is None:
        variants = list(TRAINING_VARIANTS)

    if eval_variants is None:
        eval_variants = list(EVAL_VARIANTS)

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    # Create curriculum with mission progression
    curriculum = make_curriculum(
        num_cogs=num_cogs,
        missions=missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
    )

    # Import here to avoid circular dependency
    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig

    trainer_cfg = TrainerConfig()

    # Evaluate on standard difficulty without training aids
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


def train_fast(
    num_cogs: int = 2,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Faster training variant with fewer agents and easier missions for quick iteration."""
    fast_missions = [
        "extractor_hub_30",
        "collect_resources_classic",
        "oxygen_bottleneck",
    ]

    return train(
        num_cogs=num_cogs,
        missions=fast_missions,
        policy_architecture=policy_architecture,
    )


def train_coordination(
    num_cogs: int = 4,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Training focused on coordination-heavy missions."""
    coordination_missions = [
        "divide_and_conquer",
        "go_together",
        "collect_resources_spread",
        "single_use_swarm",
    ]

    return train(
        num_cogs=num_cogs,
        missions=coordination_missions,
        policy_architecture=policy_architecture,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate the trained policy on CoGs vs Clips missions."""
    return EvaluateTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=None,  # No training aids during eval
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "extractor_hub_50",
    num_cogs: int = 4,
) -> PlayTool:
    """Interactive play with the trained policy."""
    from metta.sim.simulation_config import SimulationConfig

    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=None,  # Play without training aids
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission}_{num_cogs}cogs",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


__all__ = [
    "train",
    "train_fast",
    "train_coordination",
    "evaluate",
    "play",
]
