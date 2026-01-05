"""Core CoGs vs Clips training and evaluation recipe.

This module defines the base tooling for CoGs vs Clips training. Variant-specific
recipes should import from here and extend via custom defaults, similar to how
`recipes.experiment.abes` wraps `recipes.experiment.arena`.
"""

import logging
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER
from cogames.cogs_vs_clips.variants import VARIANTS
from metta.agent.policies.vit_size_2 import ViTSize2Config
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.nodes import default_nodes
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import CheckpointerConfig, EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import RunGate, SchedulerConfig, ScheduleRule
from metta.rl.training.teacher import TeacherConfig, apply_teacher_phase
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalWithResultTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import (
    _resolve_eval_variants,
    _resolve_mission_template,
    make_eval_suite,
    make_training_env,
)

logger = logging.getLogger(__name__)

DEFAULT_CURRICULUM_MISSIONS: list[str] = [
    "training_facility.harvest",
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
    f"hello_world{MAP_MISSION_DELIMITER}open_world",
    f"hello_world{MAP_MISSION_DELIMITER}hello_world_unclip",
    f"hello_world{MAP_MISSION_DELIMITER}oxygen_bottleneck",
    f"hello_world{MAP_MISSION_DELIMITER}energy_starved",
    f"hello_world{MAP_MISSION_DELIMITER}distant_resources",
    f"hello_world{MAP_MISSION_DELIMITER}quadrant_buildings",
    f"hello_world{MAP_MISSION_DELIMITER}single_use_swarm",
    f"hello_world{MAP_MISSION_DELIMITER}vibe_check",
    # f"machina_1{MAP_MISSION_DELIMITER}open_world",
)


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
#   -p class=checkpoint,data=s3://softmax-public/policies/...:v1 \
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
    teacher: TeacherConfig | None = None,
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
    trainer_cfg = TrainerConfig(nodes=default_nodes())
    scheduler = None
    scheduler_run_gates: list[RunGate] = []
    scheduler_rules: list[ScheduleRule] = []
    training_env_cfg = TrainingEnvironmentConfig(curriculum=curriculum)

    if teacher and teacher.enabled:
        if teacher.mode == "sliced_cloner":
            trainer_cfg.nodes["ppo_actor"].ent_coef = 0.002
        apply_teacher_phase(
            trainer_cfg=trainer_cfg,
            training_env_cfg=training_env_cfg,
            scheduler_rules=scheduler_rules,
            scheduler_run_gates=scheduler_run_gates,
            teacher_cfg=teacher,
            default_steps=teacher.steps or 300_000_000,
        )
        scheduler = SchedulerConfig(run_gates=scheduler_run_gates, rules=scheduler_rules)

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

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env_cfg,
        evaluator=evaluator_cfg,
        policy_architecture=ViTSize2Config(),
        scheduler=scheduler,
        checkpointer=CheckpointerConfig(epoch_interval=100),
    )


def train_variants(
    num_cogs: int = 4,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    teacher: TeacherConfig | None = None,
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
        teacher=teacher,
    )


def train_single_mission(
    mission: str = "training_facility.harvest",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    teacher: TeacherConfig | None = None,
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
        teacher=teacher,
    )


def evaluate(
    policy_uris: list[str] | str,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> EvalWithResultTool:
    """Evaluate policies on CoGs vs Clips missions."""
    return EvalWithResultTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uris=policy_uris,
        result_file_path="/dev/null",
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "training_facility.harvest",
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
        mission="training_facility.harvest",
        num_cogs=num_cogs,
        variants=variants,
    )


def train_coordination(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Train on coordination-heavy missions or a specific target map."""
    return train(
        num_cogs=num_cogs,
        base_missions=list(COORDINATION_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
        teacher=teacher,
    )


def train_fixed_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
    maps_cache_size: Optional[int] = 50,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Train on fixed-map CoGs vs Clips missions in one curriculum."""
    tt = train(
        num_cogs=num_cogs,
        base_missions=list(DEFAULT_CURRICULUM_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
        teacher=teacher,
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
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Train on procedural MachinaArena map missions."""
    tt = train(
        num_cogs=num_cogs,
        base_missions=list(PROC_MAP_MISSIONS),
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
        teacher=teacher,
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
