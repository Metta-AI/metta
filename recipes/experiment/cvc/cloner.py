"""Core CoGs vs Clips training and evaluation recipe.

This module defines the base tooling for CoGs vs Clips training. Variant-specific
recipes should import from here and extend via custom defaults, similar to how
`recipes.experiment.abes` wraps `recipes.experiment.arena`.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cli.mission import find_mission, parse_variants

# eval_missions.py was deleted - missions moved to integrated_evals.py
from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER, Mission, NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.cogs_vs_clips.variants import VARIANTS
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate, SchedulerConfig
from metta.rl.training.supervisor import EnvSupervisorConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)

DEFAULT_CURRICULUM_MISSIONS: list[str] = [
    "easy_hearts",
    "oxygen_bottleneck",
    "energy_starved",
]

COORDINATION_MISSIONS: list[str] = [
    "distant_resources",
    "quadrant_buildings",
    "single_use_swarm",
]

PROC_MAP_MISSIONS: tuple[str, ...] = (
    f"training_facility{MAP_MISSION_DELIMITER}harvest",
    f"training_facility{MAP_MISSION_DELIMITER}vibe_check",
    f"training_facility{MAP_MISSION_DELIMITER}repair",
    f"training_facility{MAP_MISSION_DELIMITER}easy_hearts_training_facility",
    f"hello_world{MAP_MISSION_DELIMITER}open_world",
    f"hello_world{MAP_MISSION_DELIMITER}hello_world_unclip",
    f"hello_world{MAP_MISSION_DELIMITER}oxygen_bottleneck",
    f"hello_world{MAP_MISSION_DELIMITER}energy_starved",
    f"hello_world{MAP_MISSION_DELIMITER}distant_resources",
    f"hello_world{MAP_MISSION_DELIMITER}quadrant_buildings",
    f"hello_world{MAP_MISSION_DELIMITER}single_use_swarm",
    f"hello_world{MAP_MISSION_DELIMITER}vibe_check",
    f"hello_world{MAP_MISSION_DELIMITER}easy_hearts",
    f"hello_world{MAP_MISSION_DELIMITER}easy_hearts_hello_world",
    # f"machina_1{MAP_MISSION_DELIMITER}open_world",
)


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


def _resolve_mission_template(name: str) -> Mission:
    for mission in MISSIONS:
        if mission.name == name or mission.full_name() == name:
            return mission

    if MAP_MISSION_DELIMITER not in name:
        return find_mission(name, None)

    if name.count(MAP_MISSION_DELIMITER) > 1:
        raise ValueError(f"Mission name can contain at most one '{MAP_MISSION_DELIMITER}' delimiter")

    site_name, mission_name = name.split(MAP_MISSION_DELIMITER)
    return find_mission(site_name, mission_name)


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
    variant_names: Sequence[str] | None = None,
) -> Mission:
    mission = base_mission
    variant_objects = parse_variants(list(variant_names)) if variant_names else []
    if variant_objects:
        mission = mission.with_variants(variant_objects)
    mission = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
    return mission


def make_eval_suite(
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    max_evals: Optional[int] = None,
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
            variant_names=variant_names,
        )

        env_cfg = mission.make_env()
        sim = SimulationConfig(
            suite="cogs_vs_clips",
            name=f"{mission_template.name}_{num_cogs}cogs",
            env=env_cfg,
        )
        simulations.append(sim)

    if max_evals is not None:
        logger.info(f"Limiting evaluations to {max_evals} (got {len(simulations)})")
        simulations = simulations[:max_evals]

    return simulations


def make_training_env(
    num_cogs: int = 4,
    mission: str = "easy_hearts",
    variants: Optional[Sequence[str]] = None,
) -> MettaGridConfig:
    """Create a single training environment from a mission."""
    mission_template = _resolve_mission_template(mission)

    variant_names = _normalize_variant_names(variants=variants)
    prepared_mission = _prepare_mission(
        mission_template,
        num_cogs=num_cogs,
        variant_names=variant_names,
    )
    env = prepared_mission.make_env()

    # If vibe swapping is disabled, prune stale vibe transfers to avoid invalid IDs.
    change_vibe_action = getattr(env.game.actions, "change_vibe", None)
    if change_vibe_action is not None and change_vibe_action.number_of_vibes <= 1:
        allowed_vibes = env.game.vibe_names or ["default"]
        env.game.vibe_names = list(allowed_vibes)

        chest = env.game.objects.get("chest")
        vibe_transfers = getattr(chest, "vibe_transfers", None) if chest is not None else None
        if isinstance(vibe_transfers, dict):
            allowed = set(allowed_vibes)
            chest.vibe_transfers = {vibe: transfers for vibe, transfers in vibe_transfers.items() if vibe in allowed}

    return env


def make_curriculum(
    num_cogs: int = 4,
    missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
) -> CurriculumConfig:
    """Create a curriculum for CoGs vs Clips training."""
    if missions is None:
        missions = list(DEFAULT_CURRICULUM_MISSIONS)

    all_mission_tasks = []
    for mission_name in missions:
        mission_env = make_training_env(
            num_cogs=num_cogs,
            mission=mission_name,
            variants=variants,
        )
        mission_tasks = cc.bucketed(mission_env)

        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

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


# How to submit a policy trained here to the CoGames leaderboard:
#
# uv run cogames submit \
#   -p class=mpt,kw.checkpoint_uri=s3://softmax-public/policies/...:v1.mpt \
#   -n your-policy-name-for-leaderboard \
#   --skip-validation
#
# For now we need to run --skip-validation because cogames validation
# doesn't assume the leaderboard runners get to run with the `metta` repo available,
# but in practice they do


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    mission: Optional[str] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    max_evals: Optional[int] = None,
    bc_policy_uri: Optional[str] = None,
    bc_teacher_lead_prob: float = 1.0,
    use_lp: bool = True,
) -> TrainTool:
    """Create a training tool for CoGs vs Clips."""
    training_missions = base_missions or DEFAULT_CURRICULUM_MISSIONS
    if mission is not None:
        training_missions = [mission]

    cur_alg = LearningProgressConfig() if use_lp else DiscreteRandomConfig()
    curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        missions=training_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        algorithm_config=cur_alg,
    )
    bc_policy_uri = "nim_thinky"  # av delete
    trainer_cfg = TrainerConfig(losses=LossesConfig())
    scheduler = None
    supervisor = EnvSupervisorConfig()

    if bc_policy_uri is not None:
        supervisor = EnvSupervisorConfig(policy=bc_policy_uri)

        ssc_end_step = 1_000_000_000
        trainer_cfg.losses.sliced_scripted_cloner.enabled = True
        trainer_cfg.losses.ppo_critic.sample_enabled = False
        trainer_cfg.losses.ppo_critic.train_forward_enabled = False
        trainer_cfg.losses.ppo_critic.deferred_training_start_step = ssc_end_step

        scheduler = SchedulerConfig(
            run_gates=[
                LossRunGate(loss_instance_name="ppo_critic", phase="rollout", begin_at_step=ssc_end_step),
                LossRunGate(
                    loss_instance_name="sliced_scripted_cloner",
                    phase="rollout",
                    end_at_step=ssc_end_step,
                ),
                LossRunGate(
                    loss_instance_name="sliced_scripted_cloner",
                    phase="train",
                    end_at_step=ssc_end_step,
                ),
            ],
            rules=[
                HyperUpdateRule(
                    loss_instance_name="sliced_scripted_cloner",
                    attr_path="teacher_led_proportion",
                    mode="progress",
                    style="linear",
                    start_value=0.2,
                    end_value=0.0,
                    start_agent_step=0,
                    end_agent_step=ssc_end_step,
                ),
            ],
        )

    resolved_eval_variants = _resolve_eval_variants(variants, eval_variants)
    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
        max_evals=max_evals,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
        evaluate_local=False,
    )

    tt = TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        scheduler=scheduler,
        supervisor=supervisor,
    )

    return tt


def train_variants(
    num_cogs: int = 4,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Create a training tool with curriculum tasks for all variants.

    Loads all available variants and creates a curriculum task for each one,
    merging them into a single curriculum.
    """
    if base_missions is None:
        base_missions = list(DEFAULT_CURRICULUM_MISSIONS)

    # Create tasks for each variant
    all_variant_tasks = []
    for variant in VARIANTS:
        for mission_name in base_missions:
            mission = _resolve_mission_template(mission_name)
            if not variant.compat(mission):
                continue
            mission_env = mission.make_env()
            mission_tasks = cc.bucketed(mission_env)
            all_variant_tasks.append(mission_tasks)

    # Merge all variant tasks
    merged_tasks = cc.merge(all_variant_tasks)

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=2000,
            max_slice_axes=4,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    curriculum = merged_tasks.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )

    return train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_single_mission(
    mission: str = "easy_hearts",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Train on a single mission without curriculum."""
    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )

    curriculum_cfg = cc.env_curriculum(env)

    return train(
        num_cogs=num_cogs,
        curriculum=curriculum_cfg,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
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
    mission: str = "easy_hearts",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play a single mission with a policy."""
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


def play_training_env(
    policy_uri: Optional[str] = None,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play the default training environment."""
    return play(
        policy_uri=policy_uri,
        mission="easy_hearts",
        num_cogs=num_cogs,
        variants=variants,
    )


def train_coordination(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
) -> TrainTool:
    """Train on coordination-heavy missions or a specific target map."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(COORDINATION_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
    )


def train_fixed_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
    maps_cache_size: Optional[int] = 50,
) -> TrainTool:
    """Train on fixed-map CoGs vs Clips missions in one curriculum."""
    tt = train(
        num_cogs=num_cogs,
        base_missions=list(DEFAULT_CURRICULUM_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
    )
    tt.training_env.maps_cache_size = maps_cache_size
    return tt


def train_proc_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
    maps_cache_size: Optional[int] = 50,
) -> TrainTool:
    """Train on procedural MachinaArena map missions."""
    tt = train(
        num_cogs=num_cogs,
        base_missions=list(PROC_MAP_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
    )
    tt.training_env.maps_cache_size = maps_cache_size
    return tt


__all__ = [
    "make_eval_suite",
    "make_training_env",
    "make_curriculum",
    "train",
    "train_variants",
    "train_single_mission",
    "evaluate",
    "play",
    "play_training_env",
    "train_coordination",
    "train_fixed_maps",
    "train_proc_maps",
]
