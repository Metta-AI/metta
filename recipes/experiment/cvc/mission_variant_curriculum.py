"""Mission-variant curriculum for CoGs vs Clips with learning progress.

This curriculum supports two modes:
1. All variants per mission: Creates separate curriculum tasks for each mission-variant combination
2. Shared variants: Applies the same variant(s) to all missions (or no variants for base missions)

This replaces both variants_curriculum.py and full_curriculum.py.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.evals.diagnostic_evals import (
    DIAGNOSTIC_EVALS,
)
from cogames.cogs_vs_clips.evals.eval_missions import (
    EVAL_MISSIONS,
)
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.missions import (
    HarvestMission,
    RepairMission,
    VibeCheckMission,
)
from cogames.cogs_vs_clips.variants import VARIANTS
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig
from recipes.experiment import cogs_v_clips

# Missions from eval_missions where scripted agents perform well
MISSIONS: tuple[str, ...] = (
    "go_together",  # 55.0% success, 5.22 avg reward
    "oxygen_bottleneck",  # 51.2% success, 3.02 avg reward
    "collect_resources_classic",  # 50.0% success, 4.90 avg reward
    "collect_resources_spread",  # 50.0% success, 4.45 avg reward
    "extractor_hub_70",  # 43.8% success, 1.79 avg reward
    "extractor_hub_30",
    "extractor_hub_50",
    "single_use_swarm",  # 42.5% success, 0.46 avg reward
)

# Diagnostic missions where scripted agents can get reward
DIAGNOSTIC_MISSIONS: tuple[str, ...] = (
    "diagnostic_assemble_seeded_near",
    "diagnostic_assemble_seeded_search",
    "diagnostic_extract_missing_carbon",
    "diagnostic_extract_missing_oxygen",
    "diagnostic_extract_missing_germanium",
    "diagnostic_extract_missing_silicon",
    "diagnostic_radial",
)

# Training facility missions
TRAINING_FACILITY_MISSIONS: tuple[str, ...] = (
    "harvest",
    "assemble",
    "vibe_check",
    "repair",
    "unclip_drills",
    "signs_and_portents",
)

FULL_CURRICULUM_MISSIONS: tuple[str, ...] = (
    *cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS,  # Base curriculum missions
    *MISSIONS,  # All eval missions
    # Training facility missions we currently support in this repo
    "harvest",
    "assemble",
    "vibe_check",
    "repair",
    *DIAGNOSTIC_MISSIONS,  # Diagnostic missions
)

# Create mission name mapping for eval missions and training facility missions
_MISSION_BY_NAME: dict[str, Mission] = {}
for mission in EVAL_MISSIONS:
    _MISSION_BY_NAME[mission.name] = mission

# Add training facility missions to the mapping
TRAINING_FACILITY_MISSION_OBJECTS = [
    HarvestMission,
    VibeCheckMission,
    RepairMission,
    # Note: 'easy_hearts' is intentionally omitted from the full curriculum
    # for now due to its reduced vibe action space (see comment above).
    # Note: 'assemble' is not a defined mission object - it may be resolved via make_training_env
]
for mission in TRAINING_FACILITY_MISSION_OBJECTS:
    _MISSION_BY_NAME[mission.name] = mission


def get_all_variant_names() -> list[str]:
    """Get all variant names from VARIANTS."""
    return [variant.name for variant in VARIANTS]


def _deduplicate_assembler_protocols(env: MettaGridConfig) -> None:
    """Deduplicate assembler protocols to prevent C++ config errors.

    Multiple variants can create duplicate protocols with the same vibes and min_agents.
    This function removes duplicates, keeping the first occurrence of each unique combination.
    """
    assembler = env.game.objects.get("assembler")
    if not isinstance(assembler, AssemblerConfig):
        return

    seen = set()
    deduplicated = []
    for protocol in assembler.protocols:
        # Create a key from vibes (sorted for consistency) and min_agents
        key = (tuple(sorted(protocol.vibes)), protocol.min_agents)
        if key not in seen:
            seen.add(key)
            deduplicated.append(protocol)

    assembler.protocols = deduplicated


def make_curriculum(
    base_missions: Optional[list[str] | str] = None,
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
    exclude_variants: Optional[Sequence[str] | str] = None,
    all_variants_per_mission: bool = False,
    stats_max_cap: float = 0.5,
) -> CurriculumConfig:
    """Create a mission-variant curriculum for CoGs vs Clips training with learning progress.

    Args:
        base_missions: List of mission names to include, or comma-separated string.
            If None and all_variants_per_mission=False, defaults to FULL_CURRICULUM_MISSIONS.
            Required if all_variants_per_mission=True.
        num_cogs: Number of agents per mission
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        variants: Optional mission variants to apply (only used when all_variants_per_mission=False)
        exclude_variants: Optional list of variant names to exclude, or comma-separated string
            (only used when all_variants_per_mission=True)
        all_variants_per_mission: If True, create separate tasks for each mission-variant combination.
            If False, apply the same variants to all missions (or no variants if variants=None).
        stats_max_cap: Maximum reward cap for resource stats (default: 0.5 for variants mode, 1.0 for full mode)

    Returns:
        A CurriculumConfig with learning progress algorithm
    """
    # Handle comma-separated string input (for shell compatibility)
    if isinstance(base_missions, str):
        base_missions = [m.strip() for m in base_missions.split(",") if m.strip()]

    if base_missions is None:
        if all_variants_per_mission:
            raise ValueError("base_missions must be provided when all_variants_per_mission=True")
        base_missions = list(FULL_CURRICULUM_MISSIONS)

    if not base_missions:
        raise ValueError("base_missions must be non-empty")

    # Handle comma-separated string for exclude_variants
    if isinstance(exclude_variants, str):
        exclude_variants = [v.strip() for v in exclude_variants.split(",") if v.strip()]

    all_mission_tasks = []

    if all_variants_per_mission:
        # Mode 1: Create separate tasks for each mission-variant combination
        # Get all variant names, excluding any that should be skipped
        all_variant_names = get_all_variant_names()
        if exclude_variants:
            exclude_set = set(exclude_variants)
            variant_names = [v for v in all_variant_names if v not in exclude_set]
        else:
            variant_names = all_variant_names

        for mission_name in base_missions:
            mission_template = None

            # Check if this is an eval mission
            mission_template = _MISSION_BY_NAME.get(mission_name)

            # Check if this is a diagnostic mission (class-based)
            if mission_template is None and mission_name in DIAGNOSTIC_MISSIONS:
                for mission_cls in DIAGNOSTIC_EVALS:
                    temp_mission = mission_cls()  # type: ignore[call-arg]
                    if temp_mission.name == mission_name:
                        mission_template = temp_mission
                        break

            # For each variant, create a separate curriculum task
            for variant_name in variant_names:
                try:
                    if mission_template is None:
                        # Fall back to make_training_env for standard missions
                        try:
                            mission_env = cogs_v_clips.make_training_env(
                                num_cogs=num_cogs,
                                mission=mission_name,
                                variants=[variant_name],
                            )
                        except ValueError:
                            # Skip missions that don't exist
                            continue
                        # Deduplicate assembler protocols to avoid C++ config errors
                        _deduplicate_assembler_protocols(mission_env)
                    else:
                        # Use the mission template directly (works for both eval and diagnostic missions)
                        mission = cogs_v_clips._prepare_mission(
                            mission_template,
                            num_cogs=num_cogs,
                            variant_names=[variant_name],
                        )
                        mission_env = mission.make_env()
                        # Note: Inventory clamping is handled automatically by the Inventory class

                    # Deduplicate assembler protocols to avoid C++ config errors
                    _deduplicate_assembler_protocols(mission_env)

                    # Give each environment a label so per-label rewards can be tracked in stats/W&B.
                    # Use mission:variant format to distinguish variant combinations.
                    label = f"{mission_name}:{variant_name}"
                    try:
                        mission_env.label = label  # type: ignore[attr-defined]
                    except Exception:
                        # Best-effort; if the config does not support labels, leave it as default.
                        pass

                    # Initialize stats rewards dict if needed (for bucket overrides to work)
                    if not mission_env.game.agent.rewards.stats:
                        mission_env.game.agent.rewards.stats = {}
                    # Initialize stats_max dict if needed
                    if not mission_env.game.agent.rewards.stats_max:
                        mission_env.game.agent.rewards.stats_max = {}

                    mission_tasks = cc.bucketed(mission_env)

                    # Add curriculum buckets for learning progress
                    mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])
                    # Use inventory rewards for heart which get converted to stats internally
                    mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])

                    # Add buckets for stats rewards (now supported with dict keys containing dots)
                    # Reward for gaining resources (not just having them) to avoid rewarding initial inventory
                    mission_tasks.add_bucket("game.agent.rewards.stats.carbon.gained", [0.0, 0.005, 0.01, 0.015, 0.02])
                    mission_tasks.add_bucket("game.agent.rewards.stats.oxygen.gained", [0.0, 0.005, 0.01, 0.015, 0.02])
                    mission_tasks.add_bucket(
                        "game.agent.rewards.stats.germanium.gained", [0.0, 0.005, 0.01, 0.015, 0.02]
                    )
                    mission_tasks.add_bucket("game.agent.rewards.stats.silicon.gained", [0.0, 0.005, 0.01, 0.015, 0.02])

                    # Cap resource rewards to prevent hoarding
                    mission_tasks.add_bucket("game.agent.rewards.stats_max.carbon.gained", [stats_max_cap])
                    mission_tasks.add_bucket("game.agent.rewards.stats_max.oxygen.gained", [stats_max_cap])
                    mission_tasks.add_bucket("game.agent.rewards.stats_max.germanium.gained", [stats_max_cap])
                    mission_tasks.add_bucket("game.agent.rewards.stats_max.silicon.gained", [stats_max_cap])

                    all_mission_tasks.append(mission_tasks)
                except Exception as e:
                    # Log the error but continue - some variant combinations may be incompatible
                    import logging

                    logging.warning(f"Failed to create mission-variant combination {mission_name}+{variant_name}: {e}")
                    continue

        if not all_mission_tasks:
            raise ValueError(f"No valid mission-variant combinations found for missions: {base_missions}")

    else:
        # Mode 2: Apply the same variants to all missions (or no variants)
        for mission_name in base_missions:
            mission_template = None

            # Check if this is an eval mission
            mission_template = _MISSION_BY_NAME.get(mission_name)

            # Check if this is a diagnostic mission (class-based)
            if mission_template is None and mission_name in DIAGNOSTIC_MISSIONS:
                for mission_cls in DIAGNOSTIC_EVALS:
                    temp_mission = mission_cls()  # type: ignore[call-arg]
                    if temp_mission.name == mission_name:
                        mission_template = temp_mission
                        break

            if mission_template is None:
                # Fall back to make_training_env for standard missions
                try:
                    mission_env = cogs_v_clips.make_training_env(
                        num_cogs=num_cogs,
                        mission=mission_name,
                        variants=variants,
                    )
                except ValueError:
                    # Skip missions that don't exist
                    continue
            else:
                # Use the mission template directly (works for both eval and diagnostic missions)
                variant_names = cogs_v_clips._normalize_variant_names(variants=variants)
                mission = cogs_v_clips._prepare_mission(
                    mission_template,
                    num_cogs=num_cogs,
                    variant_names=variant_names,
                )
                mission_env = mission.make_env()

            # Give each environment a label so per-label rewards can be tracked in stats/W&B.
            # Use the mission name as the label, which is stable across curriculum tasks.
            try:
                mission_env.label = mission_name  # type: ignore[attr-defined]
            except Exception:
                # Best-effort; if the config does not support labels, leave it as default.
                pass

            # Initialize stats rewards dict if needed (for bucket overrides to work)
            if not mission_env.game.agent.rewards.stats:
                mission_env.game.agent.rewards.stats = {}
            # Initialize stats_max dict if needed
            if not mission_env.game.agent.rewards.stats_max:
                mission_env.game.agent.rewards.stats_max = {}

            mission_tasks = cc.bucketed(mission_env)

            # Add curriculum buckets for learning progress
            mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])
            # Use inventory rewards for heart which get converted to stats internally
            mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])

            # Add buckets for stats rewards (now supported with dict keys containing dots)
            # Reward for gaining resources (not just having them) to avoid rewarding initial inventory
            mission_tasks.add_bucket("game.agent.rewards.stats.carbon.gained", [0.0, 0.005, 0.01, 0.015, 0.02])
            mission_tasks.add_bucket("game.agent.rewards.stats.oxygen.gained", [0.0, 0.005, 0.01, 0.015, 0.02])
            mission_tasks.add_bucket("game.agent.rewards.stats.germanium.gained", [0.0, 0.005, 0.01, 0.015, 0.02])
            mission_tasks.add_bucket("game.agent.rewards.stats.silicon.gained", [0.0, 0.005, 0.01, 0.015, 0.02])

            # Cap resource rewards to prevent hoarding
            mission_tasks.add_bucket("game.agent.rewards.stats_max.carbon.gained", [stats_max_cap])
            mission_tasks.add_bucket("game.agent.rewards.stats_max.oxygen.gained", [stats_max_cap])
            mission_tasks.add_bucket("game.agent.rewards.stats_max.germanium.gained", [stats_max_cap])
            mission_tasks.add_bucket("game.agent.rewards.stats_max.silicon.gained", [stats_max_cap])

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
    base_missions: Optional[list[str] | str] = None,
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    exclude_variants: Optional[Sequence[str] | str] = None,
    all_variants_per_mission: bool = False,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Create a training tool for CoGs vs Clips with mission-variant curriculum.

    Args:
        base_missions: List of mission names to include, or comma-separated string.
            If None and all_variants_per_mission=False, defaults to FULL_CURRICULUM_MISSIONS.
            Required if all_variants_per_mission=True.
        num_cogs: Number of agents per mission
        curriculum: Optional curriculum configuration (defaults to mission-variant curriculum)
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        variants: Optional mission variants to apply (only used when all_variants_per_mission=False)
        exclude_variants: Optional list of variant names to exclude, or comma-separated string
            (only used when all_variants_per_mission=True)
        all_variants_per_mission: If True, create separate tasks for each mission-variant combination.
            If False, apply the same variants to all missions (or no variants if variants=None).
        eval_variants: Optional mission variants to apply during evaluation
        eval_difficulty: Difficulty variant for evaluation

    Returns:
        A TrainTool configured with the mission-variant curriculum
    """
    resolved_curriculum = curriculum or make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        exclude_variants=exclude_variants,
        all_variants_per_mission=all_variants_per_mission,
        stats_max_cap=0.5 if all_variants_per_mission else 1.0,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    resolved_eval_variants = cogs_v_clips._resolve_eval_variants(variants, eval_variants)
    eval_suite = cogs_v_clips.make_eval_suite(
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


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on CoGs vs Clips missions."""
    return EvaluateTool(
        simulations=cogs_v_clips.make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play a single mission with a policy."""
    env = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )
    sim = SimulationConfig(suite="cogs_vs_clips", name=f"{mission}_{num_cogs}cogs", env=env)
    return PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    base_missions: Optional[list[str]] = None,
    run_name: Optional[str] = None,
    num_cogs: int = 4,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    variants: Optional[list[str]] = None,
    exclude_variants: Optional[list[str]] = None,
    all_variants_per_mission: bool = False,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        base_missions: Optional list of mission names to include.
            If None and all_variants_per_mission=False, defaults to FULL_CURRICULUM_MISSIONS.
            Required if all_variants_per_mission=True.
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per mission (default: 4).
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600).
        skip_git_check: Whether to skip git check (default: True).
        variants: Optional mission variants to apply (only used when all_variants_per_mission=False)
        exclude_variants: Optional list of variant names to exclude (only used when all_variants_per_mission=True)
        all_variants_per_mission: If True, create separate tasks for each mission-variant combination.
            If False, apply the same variants to all missions (or no variants if variants=None).
        additional_args: Additional arguments to pass to the training command.
    """
    if run_name is None:
        mode_str = "variants" if all_variants_per_mission else "full"
        run_name = f"mission_variant_curriculum_{mode_str}_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.mission_variant_curriculum.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        f"--heartbeat-timeout={heartbeat_timeout}",
    ]

    if base_missions:
        # Pass base_missions as comma-separated string (shell-safe format)
        missions_str = ",".join(base_missions)
        cmd.append(f"base_missions={missions_str}")

    if skip_git_check:
        cmd.append("--skip-git-check")

    if all_variants_per_mission:
        cmd.append("all_variants_per_mission=True")
        if exclude_variants:
            exclude_str = ",".join(exclude_variants)
            cmd.append(f"exclude_variants={exclude_str}")
    else:
        cmd.append("all_variants_per_mission=False")
        if variants:
            variants_str = ",".join(variants)
            cmd.append(f"variants={variants_str}")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching training job: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


__all__ = [
    "make_curriculum",
    "train",
    "evaluate",
    "play",
    "experiment",
    "FULL_CURRICULUM_MISSIONS",
    "DIAGNOSTIC_MISSIONS",
    "TRAINING_FACILITY_MISSIONS",
]
