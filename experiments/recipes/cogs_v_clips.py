"""Core CoGs vs Clips training and evaluation recipe.

This module defines the base tooling for CoGs vs Clips training. Variant-specific
recipes should import from here and extend via custom defaults, similar to how
`experiments.recipes.abes` wraps `experiments.recipes.arena`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cli.mission import parse_variants
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

DEFAULT_CURRICULUM_MISSIONS: tuple[str, ...] = (
    "extractor_hub_30",
    "extractor_hub_50",
    "collect_resources_classic",
    "collect_resources_spread",
    "oxygen_bottleneck",
    "energy_starved",
)

SMALL_MAP_MISSIONS: tuple[str, ...] = (
    "extractor_hub_30",
    "collect_resources_classic",
    "oxygen_bottleneck",
)

MEDIUM_MAP_MISSIONS: tuple[str, ...] = (
    "extractor_hub_50",
    "collect_resources_spread",
    "energy_starved",
)

LARGE_MAP_MISSIONS: tuple[str, ...] = (
    "extractor_hub_70",
    "collect_far",
    "divide_and_conquer",
)

COORDINATION_MISSIONS: tuple[str, ...] = (
    "go_together",
    "divide_and_conquer",
    "collect_resources_spread",
)

_MISSION_BY_NAME: dict[str, Mission] = {
    mission.name: mission for mission in EVAL_MISSIONS
}


def _normalize_variant_names(
    *,
    initial: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> list[str]:
    names: list[str] = []
    for source in (initial, variants):
        if not source:
            continue
        for name in source:
            if name not in names:
                names.append(name)
    return names


def _parse_variant_objects(names: Sequence[str] | None) -> list[MissionVariant]:
    if not names:
        return []
    return parse_variants(list(names))


def _resolve_eval_variants(
    train_variants: Optional[Sequence[str]],
    eval_variants: Optional[Sequence[str]],
) -> Optional[list[str]]:
    if eval_variants is not None:
        return list(eval_variants)
    if train_variants is not None:
        return list(train_variants)
    return None


def _prepare_mission(
    base_mission: Mission,
    *,
    num_cogs: int,
    variant_objects: Sequence[MissionVariant] | None = None,
) -> Mission:
    mission = base_mission
    if variant_objects:
        mission = mission.with_variants(list(variant_objects))
    mission = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
    return mission


def make_eval_suite(
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> list[SimulationConfig]:
    """Create a suite of evaluation simulations from CoGames missions.

    Args:
        num_cogs: Number of agents per mission (1, 2, 4, or 8)
        difficulty: Difficulty variant to apply (e.g., "standard", "hard", "story_mode")
        subset: Optional list of mission names to include (defaults to all)
        variants: Additional mission variants to apply (lonely_heart, heart_chorus, ...)

    Returns:
        A list of SimulationConfig objects ready for evaluation.
    """
    if subset:
        missions = [m for m in EVAL_MISSIONS if m.name in subset]
    else:
        missions = EVAL_MISSIONS

    variant_names = _normalize_variant_names(
        initial=[difficulty] if difficulty else None,
        variants=variants,
    )

    variant_objects = _parse_variant_objects(variant_names)

    simulations: list[SimulationConfig] = []
    for mission_template in missions:
        if num_cogs == 1 and mission_template.name in {
            "go_together",
            "single_use_swarm",
        }:
            continue

        mission = _prepare_mission(
            mission_template,
            num_cogs=num_cogs,
            variant_objects=variant_objects,
        )

        env_cfg = mission.make_env()
        sim = SimulationConfig(
            suite="cogs_vs_clips",
            name=f"{mission_template.name}_{num_cogs}cogs",
            env=env_cfg,
        )
        simulations.append(sim)

    return simulations


def make_training_env(
    num_cogs: int = 4,
    mission_name: str = "extractor_hub_30",
    variants: Optional[Sequence[str]] = None,
) -> MettaGridConfig:
    """Create a single training environment from a mission."""
    mission_template = _MISSION_BY_NAME.get(mission_name)
    if mission_template is None:
        raise ValueError(f"Mission '{mission_name}' not found in EVAL_MISSIONS")

    variant_names = _normalize_variant_names(variants=variants)
    variant_objects = _parse_variant_objects(variant_names)
    mission = _prepare_mission(
        mission_template,
        num_cogs=num_cogs,
        variant_objects=variant_objects,
    )
    env = mission.make_env()

    # Guard against upstream modifiers pushing limits beyond supported bounds.
    energy_limit = env.game.agent.resource_limits.get("energy")
    if isinstance(energy_limit, int) and energy_limit > 255:
        env.game.agent.resource_limits["energy"] = 255
        if env.game.agent.initial_inventory.get("energy", 0) > 255:
            env.game.agent.initial_inventory["energy"] = 255

    # If vibe swapping is disabled, prune stale vibe transfers to avoid invalid IDs.
    change_vibe_action = getattr(env.game.actions, "change_vibe", None)
    if change_vibe_action is not None and change_vibe_action.number_of_vibes <= 1:
        allowed_vibes = set(env.game.vibe_names or [])
        if not allowed_vibes:
            allowed_vibes = {"default"}
            env.game.vibe_names = ["default"]

        chest = env.game.objects.get("chest")
        if chest is not None and hasattr(chest, "vibe_transfers"):
            try:
                chest.vibe_transfers = {
                    vibe: transfers
                    for vibe, transfers in chest.vibe_transfers.items()
                    if vibe in allowed_vibes
                }
            except Exception:
                pass

    return env


def make_curriculum(
    num_cogs: int = 4,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
) -> CurriculumConfig:
    """Create a curriculum for CoGs vs Clips training."""
    if base_missions is None:
        base_missions = list(DEFAULT_CURRICULUM_MISSIONS)

    all_mission_tasks = []
    for mission_name in base_missions:
        mission_env = make_training_env(
            num_cogs=num_cogs,
            mission_name=mission_name,
            variants=variants,
        )
        mission_tasks = cc.bucketed(mission_env)

        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])
        mission_tasks.add_bucket(
            "game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0]
        )

        all_mission_tasks.append(mission_tasks)

    merged_tasks = cc.merge(all_mission_tasks)

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=2000,
            max_slice_axes=4,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return merged_tasks.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Create a training tool for CoGs vs Clips."""
    resolved_curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    resolved_eval_variants = _resolve_eval_variants(variants, eval_variants)
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def train_single_mission(
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on a single mission without curriculum."""
    env = make_training_env(
        num_cogs=num_cogs,
        mission_name=mission_name,
        variants=variants,
    )

    curriculum_cfg = cc.env_curriculum(env)

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    resolved_eval_variants = _resolve_eval_variants(variants, eval_variants)
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum_cfg),
        evaluator=EvaluatorConfig(simulations=eval_suite),
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on CoGs vs Clips missions."""
    return EvaluateTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play a single mission with a policy."""
    env = make_training_env(
        num_cogs=num_cogs,
        mission_name=mission_name,
        variants=variants,
    )
    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission_name}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_training_env(
    policy_uri: Optional[str] = None,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play the default training environment."""
    return play(
        policy_uri=policy_uri,
        mission_name="extractor_hub_30",
        num_cogs=num_cogs,
        variants=variants,
    )


# Convenience entrypoints ----------------------------------------------------


def train_small_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on small maps (30x30, classic layouts)."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(SMALL_MAP_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_medium_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on medium maps (50x50, spread layouts)."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(MEDIUM_MAP_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_large_maps(
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on large maps with more agents."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(LARGE_MAP_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_coordination(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on missions emphasizing multi-agent coordination."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(COORDINATION_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


__all__ = [
    "make_eval_suite",
    "make_training_env",
    "make_curriculum",
    "train",
    "train_single_mission",
    "evaluate",
    "play",
    "play_training_env",
    "train_small_maps",
    "train_medium_maps",
    "train_large_maps",
    "train_coordination",
]
