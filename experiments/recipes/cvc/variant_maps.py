"""CVC curriculum combining all map sizes and variants.

This module creates a comprehensive curriculum that generates individual tasks
for each combination of map size, mission, and behavioral variant. Each task
is labeled for tracking learning progress across the full combinatorial space.

Uses a bucketed curriculum approach where each (mission, variant) combination
is a separate bucket. This allows each task to have its own action space
configuration (e.g., different numbers of vibes) while still being part of
a unified curriculum.

Example labels:
  - small_extractor_hub_30_lonely_heart
  - medium_collect_resources_spread_pack_rat
  - large_divide_and_conquer_heart_chorus
"""

from __future__ import annotations

from typing import Optional, Sequence

from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from experiments.recipes.cogs_v_clips import (
    LARGE_MAP_MISSIONS,
    MEDIUM_MAP_MISSIONS,
    SMALL_MAP_MISSIONS,
    make_eval_suite,
    make_training_env,
)
from experiments.recipes.cvc.variants import CORE_VARIANTS
import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig


# Mission registry
_MISSION_BY_NAME: dict[str, Mission] = {
    mission.name: mission for mission in EVAL_MISSIONS
}


def make_curriculum(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    include_small_maps: bool = True,
    include_medium_maps: bool = True,
    include_large_maps: bool = False,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create curriculum with all map-variant combinations.

    This uses a bucketed curriculum approach where each unique (mission, variant)
    combination is a separate task with its own action space configuration.
    Tasks are labeled for tracking (e.g., "small_extractor_hub_30_lonely_heart").

    Args:
        num_cogs: Number of agents
        variants: List of variants to include (default: all 4 core variants)
        include_small_maps: Include 30x30 maps
        include_medium_maps: Include 50x50 maps
        include_large_maps: Include 70x70+ maps (requires num_cogs >= 8)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        algorithm_config: Custom curriculum algorithm config

    Returns:
        CurriculumConfig with comprehensive map-variant task generator
    """
    # Validate large map configuration
    if include_large_maps and num_cogs < 8:
        raise ValueError(
            f"Large maps require at least 8 agents, got {num_cogs}. "
            "Set num_cogs=8 or include_large_maps=False"
        )

    # Use all core variants by default
    resolved_variants = list(variants) if variants else list(CORE_VARIANTS)

    # Build list of all task combinations
    tasks_to_create: list[tuple[str, str, str]] = []  # (size, mission, variant)

    if include_small_maps:
        for mission in SMALL_MAP_MISSIONS:
            for variant in resolved_variants:
                tasks_to_create.append(("small", mission, variant))

    if include_medium_maps:
        for mission in MEDIUM_MAP_MISSIONS:
            for variant in resolved_variants:
                tasks_to_create.append(("medium", mission, variant))

    if include_large_maps:
        for mission in LARGE_MAP_MISSIONS:
            for variant in resolved_variants:
                tasks_to_create.append(("large", mission, variant))

    # Create a bucketed curriculum where each task is a separate bucket
    # This allows each task to have its own action space configuration
    all_task_generators = []

    for size, mission, variant in tasks_to_create:
        # Create environment for this specific combination
        env = make_training_env(
            num_cogs=num_cogs,
            mission=mission,
            variants=[variant],
        )

        # Set label for curriculum tracking
        label = f"{size}_{mission}_{variant}"
        env.label = label

        # Normalize action space: ensure ALL tasks use the full vibe set
        # This is required for consistent action counts across all tasks
        from cogames.cogs_vs_clips import vibes

        change_vibe = getattr(env.game.actions, "change_vibe", None)
        if change_vibe is not None:
            # Always keep enabled and use full vibe list for consistent action space
            change_vibe.enabled = True
            change_vibe.number_of_vibes = len(vibes.VIBES)

            # Restore full vibe names (neutral_faced reduces this to ['default'])
            env.game.vibe_names = [vibe.name for vibe in vibes.VIBES]

            # Note: For neutral_faced, this means the vibe action exists but
            # assembler protocols still only accept 'default' vibe, so it's
            # effectively a no-op from a gameplay perspective

        # Create a single-task generator for this combination
        task_gen = cc.single_task(env)
        all_task_generators.append(task_gen)

    # Merge all task generators into a unified curriculum
    merged_tasks = cc.merge(all_task_generators)

    # Default LP algorithm config
    # Note: With duplicate tasks (many task_ids mapping to same config via merge),
    # we need fewer samples per task_id for LP calculation since tasks get evicted frequently
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            num_active_tasks=256,
            slow_timescale_factor=0.2,
            rand_task_rate=0.01,
            exploration_bonus=0.1,
            min_samples_for_lp=2,  # Reduced from 10 to work with duplicate tasks
            enable_detailed_slice_logging=enable_detailed_slice_logging,
            lp_score_temperature=0.0,
            z_score_amplification=50.0,
            show_curriculum_troubleshooting_logging=True,
            early_progress_amplification=0.5,
            max_slice_axes=4,
        )

    curriculum_config = merged_tasks.to_curriculum(
        num_active_tasks=256,
        algorithm_config=algorithm_config,
    )

    # Set eviction threshold to allow LP scores to stabilize before eviction
    # With min_samples_for_lp=2, tasks start calculating LP after 2 completions
    # We need additional completions for LP to influence sampling distribution
    # Setting to 5 gives 3 completions where LP actively guides task selection
    curriculum_config.min_presentations_for_eviction = 5

    return curriculum_config


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    include_small_maps: bool = True,
    include_medium_maps: bool = True,
    include_large_maps: bool = False,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on comprehensive map-variant curriculum.

    This creates a curriculum with individual tasks for each combination of:
    - Map size (small/medium/large)
    - Mission within that size
    - Behavioral variant

    Each task is labeled (e.g., "small_extractor_hub_30_lonely_heart") for
    tracking learning progress across the full combinatorial space.

    Args:
        num_cogs: Number of agents (use 8 for large maps)
        variants: List of variants to train on (default: all 4 core variants)
        include_small_maps: Include 30x30 maps (3 missions)
        include_medium_maps: Include 50x50 maps (3 missions)
        include_large_maps: Include 70x70+ maps (3 missions, requires 8 agents)
        curriculum: Custom curriculum config (overrides other params)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        eval_variants: Variants to evaluate on (default: same as training)
        eval_difficulty: Eval difficulty level

    Returns:
        TrainTool configured for variant-map curriculum

    Example:
        # Train on small + medium maps with all 4 variants (24 tasks total)
        uv run ./tools/run.py experiments.recipes.cvc.variant_maps.train \\
            run=variant_maps num_cogs=4

        # Train on all maps including large (72 tasks total)
        uv run ./tools/run.py experiments.recipes.cvc.variant_maps.train \\
            run=all_maps num_cogs=8 include_large_maps=True

        # Train on small maps with 2 specific variants (6 tasks)
        uv run ./tools/run.py experiments.recipes.cvc.variant_maps.train \\
            run=small_custom 'variants=["lonely_heart","heart_chorus"]'
    """
    resolved_curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        variants=variants,
        include_small_maps=include_small_maps,
        include_medium_maps=include_medium_maps,
        include_large_maps=include_large_maps,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    # Resolve eval variants
    if eval_variants is None:
        eval_variants = list(variants) if variants else list(CORE_VARIANTS)

    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=eval_variants,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def train_all_variants_all_sizes(
    num_cogs: int = 8,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on all combinations: small, medium, large maps × 4 variants = 72 tasks.

    Requires 8 agents for large maps.
    """
    return train(
        num_cogs=num_cogs,
        include_small_maps=True,
        include_medium_maps=True,
        include_large_maps=True,
        eval_difficulty=eval_difficulty,
    )


def train_small_medium_all_variants(
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on small + medium maps × 4 variants = 24 tasks."""
    return train(
        num_cogs=num_cogs,
        include_small_maps=True,
        include_medium_maps=True,
        include_large_maps=False,
        eval_difficulty=eval_difficulty,
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "extractor_hub_30",
    variant: str = "lonely_heart",
    num_cogs: int = 4,
) -> PlayTool:
    """Play a specific mission-variant combination.

    Args:
        policy_uri: Path to policy checkpoint
        mission: Mission name (e.g., "extractor_hub_30")
        variant: Variant name (one of the 4 core variants)
        num_cogs: Number of agents
    """
    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=[variant],
    )

    return PlayTool(
        sim=SimulationConfig(
            suite="cvc_variant_maps",
            name=f"{mission}_{variant}",
            env=env,
        ),
        policy_uri=policy_uri,
    )


__all__ = [
    "make_curriculum",
    "train",
    "train_all_variants_all_sizes",
    "train_small_medium_all_variants",
    "play",
]
