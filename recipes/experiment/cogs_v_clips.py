"""Core CoGs vs Clips training and evaluation recipe.

This module defines the base tooling for CoGs vs Clips training. Variant-specific
recipes should import from here and extend via custom defaults, similar to how
`recipes.experiment.abes` wraps `recipes.experiment.arena`.
"""

import itertools
import logging
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cli.mission import find_mission, parse_variants

# eval_missions.py was deleted - missions moved to integrated_evals.py
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER, Mission, NumCogsVariant
from cogames.cogs_vs_clips.missions import get_core_missions
from cogames.cogs_vs_clips.variants import VARIANTS
from devops.stable.registry import ci_job, stable_job
from devops.stable.runner import AcceptanceCriterion
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.common.wandb.context import WandbConfig
from metta.rl.nodes.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import CheckpointerConfig, EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import LossRunGate, SchedulerConfig, ScheduleRule
from metta.rl.training.teacher import TeacherConfig, apply_teacher_phase
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import ParameterSpec, make_sweep
from metta.sweep.core import SweepParameters as SP
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.request_remote_eval import RequestRemoteEvalTool
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)

# Single canonical curriculum list (fixed + procedural)
DEFAULT_CURRICULUM_MISSIONS: list[str] = [
    # Core hello_world missions
    "hello_world.oxygen_bottleneck",
    "hello_world.energy_starved",
    "hello_world.distant_resources",
    "hello_world.quadrant_buildings",
    "hello_world.single_use_swarm",
    # Additional fixed/procedural maps
    "hello_world.hello_world_unclip",
    "hello_world.open_world",
    # Machina maps
    "machina_1.open_world",
    "machina_1.balanced_corners",
]

COORDINATION_MISSIONS: list[str] = [
    "distant_resources",
    "quadrant_buildings",
    "single_use_swarm",
]


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
    for mission in get_core_missions():
        if mission.name == name or mission.full_name() == name:
            return mission

    if MAP_MISSION_DELIMITER not in name:
        return find_mission(name, None, include_evals=True)

    if name.count(MAP_MISSION_DELIMITER) > 1:
        raise ValueError(f"Mission name can contain at most one '{MAP_MISSION_DELIMITER}' delimiter")

    site_name, mission_name = name.split(MAP_MISSION_DELIMITER)
    return find_mission(site_name, mission_name, include_evals=True)


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
    missions: Optional[Sequence[Mission]] = None,
) -> list[SimulationConfig]:
    """Create a suite of evaluation simulations from CoGames missions.

    Args:
        num_cogs: Number of agents per mission (1, 2, 4, or 8)
        difficulty: Difficulty variant to apply (e.g., "standard", "hard", "story_mode")
        subset: Optional list of mission names to include (defaults to all)
        variants: Additional mission variants to apply (heart_chorus, ...)

    Returns:
        A list of SimulationConfig objects ready for evaluation.
    """
    if missions is not None:
        eval_missions = list(missions)
    else:
        from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS

        eval_missions = list(INTEGRATED_EVAL_MISSIONS)
    if subset:
        eval_missions = [m for m in eval_missions if m.name in subset]

    variant_names = _normalize_variant_names(
        initial=[difficulty] if difficulty else None,
        variants=variants,
    )

    simulations: list[SimulationConfig] = []
    for mission_template in eval_missions:
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
    mission: str = "training_facility.harvest",
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
    if change_vibe_action is not None and len(change_vibe_action.vibes) <= 1:
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
    difficulty: str | None = None,
    dr_variants: int = 0,
    dr_rewards: bool = True,
    dr_misc: bool = False,
) -> CurriculumConfig:
    """Create a curriculum for CoGs vs Clips training."""
    if missions is None:
        missions = list(DEFAULT_CURRICULUM_MISSIONS)

    all_mission_tasks = []
    for mission_name in missions:
        mission_template = _resolve_mission_template(mission_name)

        # Determine which variant sets to use for bucketing
        if variants is None:
            available = [v.name for v in VARIANTS if v.compat(mission_template)]
            if dr_variants == 0:
                variant_sets: list[list[str] | None] = [None] + [[v] for v in available]
            else:
                max_k = min(dr_variants, len(available))
                variant_sets = [
                    list(combo) if combo else None
                    for k in range(max_k + 1)
                    for combo in itertools.combinations(available, k)
                ]
        else:
            available = [
                name for name in variants if any(v.name == name and v.compat(mission_template) for v in VARIANTS)
            ]
            if dr_variants == 0:
                variant_sets = [available] if available else []
            else:
                max_k = min(dr_variants, len(available))
                variant_sets = [
                    list(combo) if combo else None
                    for k in range(max_k + 1)
                    for combo in itertools.combinations(available, k)
                ]

        for variant_set in variant_sets:
            applied_variants = _normalize_variant_names(
                initial=[difficulty] if difficulty else None,
                variants=variant_set,
            )
            mission_env = make_training_env(
                num_cogs=num_cogs,
                mission=mission_name,
                variants=applied_variants or None,
            )
            mission_env.game.global_obs.goal_obs = True
            mission_tasks = cc.bucketed(mission_env)
            mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

            if dr_rewards:
                mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited_by_agent", [0, 1, 5, 10])
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0, 1, 5, 10])
                resources = ["carbon", "oxygen", "germanium", "silicon"]
                for resource in resources:
                    mission_tasks.add_bucket(f"game.agent.rewards.inventory.{resource}", [0.0, 0.01, 0.1, 1])
                equipment = ["scrambler", "modulator", "decoder", "resonator"]
                for item in equipment:
                    mission_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0.0, 0.1, 1.0, 10.0])

            if dr_misc:
                mission_tasks.add_bucket("game.agent.inventory_regen_amounts.energy", [0, 1, 2])
                mission_tasks.add_bucket("game.actions.move.consumed_resources.energy", [1, 2, 3])
                mission_tasks.add_bucket("game.agent.resource_limits.cargo.limit", [25, 50, 100])
                mission_tasks.add_bucket("game.agent.resource_limits.energy.limit", [50, 75, 100])
                mission_tasks.add_bucket("game.clipper.clip_period", [0, 25, 50])
                mission_tasks.add_bucket("game.inventory_regen_interval", [0, 1, 2])
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
    train_difficulty: str | None = None,
    eval_mission_source: str = "integrated_evals",
    max_evals: Optional[int] = None,
    teacher: TeacherConfig | None = None,
    use_lp: bool = True,
    dr_variants: int = 0,
    dr_rewards: bool = True,
    dr_misc: bool = False,
    maps_cache_size: Optional[int] = 30,
) -> TrainTool:
    """Create a training tool for CoGs vs Clips."""
    training_missions = base_missions or DEFAULT_CURRICULUM_MISSIONS
    if mission is not None:
        training_missions = [mission]

    if train_difficulty is None:
        train_difficulty = eval_difficulty

    cur_alg = LearningProgressConfig() if use_lp else DiscreteRandomConfig()
    curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        missions=training_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        difficulty=train_difficulty,
        algorithm_config=cur_alg,
        dr_variants=dr_variants,
        dr_rewards=dr_rewards,
        dr_misc=dr_misc,
    )

    trainer_cfg = TrainerConfig(losses=LossesConfig())

    resolved_eval_variants = _resolve_eval_variants(variants, eval_variants)
    eval_missions: Optional[list[Mission]] = None
    if eval_mission_source == "training_missions":
        eval_missions = [_resolve_mission_template(name) for name in training_missions]
    elif eval_mission_source != "integrated_evals":
        raise ValueError(f"Unknown eval_mission_source: {eval_mission_source}")

    eval_suite = make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
        max_evals=max_evals,
        missions=eval_missions,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
        epoch_interval=150,
    )

    tt = TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
    )

    if maps_cache_size is not None:
        tt.training_env.maps_cache_size = maps_cache_size

    scheduler_run_gates: list[LossRunGate] = []
    scheduler_rules: list[ScheduleRule] = []

    if teacher and teacher.enabled:
        apply_teacher_phase(
            trainer_cfg=tt.trainer,
            training_env_cfg=tt.training_env,
            scheduler_rules=scheduler_rules,
            scheduler_run_gates=scheduler_run_gates,
            teacher_cfg=teacher,
        )

    tt.scheduler = SchedulerConfig(run_gates=scheduler_run_gates, rules=scheduler_rules)

    return tt


def train_variants(
    num_cogs: int = 4,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    train_difficulty: str | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Create a training tool with curriculum tasks for all variants.

    Loads all available variants and creates a curriculum task for each one,
    merging them into a single curriculum.
    """
    if base_missions is None:
        base_missions = list(DEFAULT_CURRICULUM_MISSIONS)

    if train_difficulty is None:
        train_difficulty = eval_difficulty

    # Create tasks for each variant
    all_variant_tasks = []
    for variant in VARIANTS:
        for mission_name in base_missions:
            mission = _resolve_mission_template(mission_name)
            if not variant.compat(mission):
                continue
            if train_difficulty:
                mission = mission.with_variants(parse_variants([train_difficulty]))
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
    train_difficulty: str | None = None,
    teacher: TeacherConfig | None = None,
    maps_cache_size: Optional[int] = 30,
) -> TrainTool:
    """Train on a single mission without curriculum."""
    if train_difficulty is None:
        train_difficulty = eval_difficulty

    training_variants = _normalize_variant_names(
        initial=[train_difficulty] if train_difficulty else None,
        variants=variants,
    )

    env = make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=training_variants or None,
    )

    curriculum_cfg = cc.env_curriculum(env)

    return train(
        num_cogs=num_cogs,
        curriculum=curriculum_cfg,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        train_difficulty=train_difficulty,
        teacher=teacher,
        maps_cache_size=maps_cache_size,
    )


def evaluate(
    policy_uris: list[str] | str,
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


def evaluate_remote(
    policy_uri: str | None = None,
    policy_version_id: str | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> RequestRemoteEvalTool:
    """Evaluate policies on CoGs vs Clips missions remotely."""
    return RequestRemoteEvalTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uri=policy_uri,
        policy_version_id=policy_version_id,
        push_metrics_to_wandb=False,
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


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
) -> TrainTool:
    """Train with heart_chorus baked in (CLI-friendly for sweeps)."""
    base_variants = ["heart_chorus"]
    if variants:
        for v in variants:
            if v not in base_variants:
                base_variants.append(v)

    tool = train(
        num_cogs=num_cogs,
        base_missions=None,
        variants=base_variants,
        eval_variants=eval_variants or base_variants,
        eval_difficulty=eval_difficulty,
        mission=mission,
    )
    # Sweep-friendly default (kept consistent with the sweep search space).
    tool.trainer.total_timesteps = 1_000_000_000
    return tool


def evaluate_stub(*args, **kwargs) -> StubTool:
    """No-op evaluator for sweeps (avoids dispatching eval jobs)."""
    return StubTool()


def get_cvc_sweep_search_space() -> dict[str, ParameterSpec]:
    """Shared sweep parameters for CvC-style PPO + schedulefree runs."""
    return {
        # Optimizer
        **SP.param(
            "trainer.optimizer.learning_rate",
            D.LOG_NORMAL,
            min=1e-3,
            max=3e-2,
            search_center=1e-2,
        ),
        **SP.param(
            "trainer.optimizer.eps",
            D.LOG_NORMAL,
            min=1e-8,
            max=5e-5,
            search_center=1e-6,
        ),
        **SP.param(
            "trainer.optimizer.warmup_steps",
            D.INT_UNIFORM,
            min=0,
            max=10_000,
            search_center=2300,
        ),
        **SP.param(
            "trainer.optimizer.weight_decay",
            D.LOG_NORMAL,
            min=1e-5,
            max=1e-1,
            search_center=1e-2,
        ),
        **SP.param(
            "trainer.optimizer.momentum",
            D.UNIFORM,
            min=0.7,
            max=0.99,
            search_center=0.9,
        ),
        # PPO
        **SP.param(
            "trainer.losses.ppo_actor.clip_coef",
            D.UNIFORM,
            min=0.05,
            max=0.4,
            search_center=0.26,
        ),
        **SP.param(
            "trainer.advantage.gae_lambda",
            D.UNIFORM,
            min=0.8,
            max=0.995,
            search_center=0.97,
        ),
        **SP.param(
            "trainer.losses.ppo_critic.vf_coef",
            D.UNIFORM,
            min=0.1,
            max=2.0,
            search_center=0.75,
        ),
        **SP.param(
            "trainer.losses.ppo_actor.ent_coef",
            D.LOG_NORMAL,
            min=0.001,
            max=0.1,
            search_center=0.025,
        ),
        **SP.param(
            "trainer.advantage.gamma",
            D.UNIFORM,
            min=0.95,
            max=0.9995,
            search_center=0.99,
        ),
        **SP.categorical(
            "trainer.losses.ppo_critic.vf_clip_coef",
            choices=[0.0, 0.1, 0.2, 0.3],
        ),
        **SP.categorical(
            "trainer.sampling.method",
            choices=["sequential", "prioritized"],
        ),
        **SP.param(
            "trainer.sampling.prio_alpha",
            D.UNIFORM,
            min=0.0,
            max=1.0,
            search_center=0.4,
        ),
        **SP.param(
            "trainer.sampling.prio_beta0",
            D.UNIFORM,
            min=0.2,
            max=1.0,
            search_center=0.6,
        ),
        **SP.categorical(
            "policy_architecture.core_resnet_layers",
            choices=[1, 2, 3, 4],
        ),
        **SP.categorical(
            "policy_architecture.latent_dim",
            choices=[64, 96, 128],
        ),
        **SP.categorical(
            "policy_architecture.actor_hidden",
            choices=[256, 384, 512],
        ),
        **SP.categorical(
            "policy_architecture.core_num_heads",
            choices=[2, 4, 6],
        ),
        **SP.categorical(
            "policy_architecture.critic_hidden",
            choices=[512, 768, 1024],
        ),
        **SP.categorical(
            "policy_architecture.core_num_latents",
            choices=[12, 16, 20],
        ),
    }


def sweep(
    sweep_name: str,
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
    max_trials: int = 80,
    num_parallel_trials: int = 4,
) -> SweepTool:
    """Hyperparameter sweep targeting train_sweep (heart_chorus baked in)."""

    search_space = get_cvc_sweep_search_space()

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.cogs_v_clips",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        metric_key="env_game/assembler.heart.created",
        search_space=search_space,
        cost_key="metric/total_time",
        max_trials=max_trials,
        num_parallel_trials=num_parallel_trials,
    )


@ci_job(timeout_s=240)
def train_ci() -> TrainTool:
    """Minimal CvC train for CI smoke test."""
    env = make_training_env(
        num_cogs=2,
        mission="training_facility.harvest",
        variants=["heart_chorus"],
    )
    curriculum_cfg = cc.env_curriculum(env)
    return TrainTool(
        trainer=TrainerConfig(
            total_timesteps=64,
            minibatch_size=8,
            batch_size=64,
            bptt_horizon=8,
            update_epochs=1,
        ),
        training_env=TrainingEnvironmentConfig(
            curriculum=curriculum_cfg,
            forward_pass_minibatch_target_size=8,
            vectorization="serial",
            auto_workers=False,
            num_workers=1,
            async_factor=1,
            maps_cache_size=4,
        ),
        evaluator=EvaluatorConfig(epoch_interval=0, evaluate_local=False, evaluate_remote=False),
        checkpointer=CheckpointerConfig(epoch_interval=1),
        wandb=WandbConfig.Off(),
    )


@ci_job(timeout_s=120)
def play_ci() -> PlayTool:
    """CvC play test with random policy."""
    env = make_training_env(num_cogs=2, mission="training_facility.harvest")
    sim = SimulationConfig(suite="cogs_vs_clips", name="harvest_ci", env=env)
    return PlayTool(sim=sim, max_steps=10, render="log", open_browser_on_start=False)


@stable_job(
    remote_gpus=1,
    remote_nodes=1,
    timeout_s=43200,
    # NOTE: as of 12/17/2025, this sometimes fails to meet 30,000
    # See https://wandb.ai/metta-research/metta/runs/runner.all.2025.12.17-024414-cogs_v_clips.train_200ep/overview?nw=nwusernishadsingh
    acceptance=[AcceptanceCriterion(metric="overview/sps", threshold=29000)],
)
def train_200ep() -> TrainTool:
    """CvC 200 epochs (~105M timesteps)."""
    tool = train(num_cogs=4, variants=["heart_chorus"])
    tool.trainer.total_timesteps = 200 * 524288
    return tool


@stable_job(
    remote_gpus=4,
    remote_nodes=4,
    timeout_s=172800,
    acceptance=[AcceptanceCriterion(metric="overview/sps", threshold=80000)],
)
def train_2b() -> TrainTool:
    """CvC multi GPU - 2B timesteps."""
    tool = train(num_cogs=4, variants=["heart_chorus"])
    tool.trainer.total_timesteps = 2_000_000_000
    return tool
