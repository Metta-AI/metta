"""Variants curriculum for CoGs vs Clips with learning progress.

This curriculum takes a set of maps and creates a curriculum over every possible variant
in variants.py. For each map, it creates separate curriculum tasks for each variant.
"""

from __future__ import annotations

import json
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
    AssembleMission,
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
from recipes.experiment import cogs_v_clips
import metta.tools as tools

# Create mission name mapping for eval missions and training facility missions
_MISSION_BY_NAME: dict[str, Mission] = {}
for mission in EVAL_MISSIONS:
    _MISSION_BY_NAME[mission.name] = mission

# Add training facility missions to the mapping
TRAINING_FACILITY_MISSION_OBJECTS = [
    HarvestMission,
    AssembleMission,
    VibeCheckMission,
    RepairMission,
]
for mission in TRAINING_FACILITY_MISSION_OBJECTS:
    _MISSION_BY_NAME[mission.name] = mission

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


def get_all_variant_names() -> list[str]:
    """Get all variant names from VARIANTS."""
    return [variant.name for variant in VARIANTS]


def make_curriculum(
    base_missions: list[str],
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    exclude_variants: Optional[Sequence[str]] = None,
) -> CurriculumConfig:
    """Create a variants curriculum for CoGs vs Clips training with learning progress.

    This curriculum takes a set of maps and creates a curriculum over every possible variant.
    For each map, it creates separate curriculum tasks for each variant individually.

    Args:
        base_missions: List of mission names to include (required)
        num_cogs: Number of agents per mission
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        exclude_variants: Optional list of variant names to exclude from the curriculum

    Returns:
        A CurriculumConfig with learning progress algorithm
    """
    if not base_missions:
        raise ValueError("base_missions must be provided and non-empty")

    # Get all variant names, excluding any that should be skipped
    all_variant_names = get_all_variant_names()
    if exclude_variants:
        exclude_set = set(exclude_variants)
        variant_names = [v for v in all_variant_names if v not in exclude_set]
    else:
        variant_names = all_variant_names

    all_mission_tasks = []
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
                else:
                    # Use the mission template directly (works for both eval and diagnostic missions)
                    mission = cogs_v_clips._prepare_mission(
                        mission_template,
                        num_cogs=num_cogs,
                        variant_names=[variant_name],
                    )
                    mission_env = mission.make_env()
                    cogs_v_clips._clamp_agent_inventory(mission_env)

                # Give each environment a label so per-label rewards can be tracked in stats/W&B.
                # Use mission_name + variant_name as the label to distinguish variant combinations.
                label = f"{mission_name}_{variant_name}"
                try:
                    mission_env.label = label  # type: ignore[attr-defined]
                except Exception:
                    # Best-effort; if the config does not support labels, leave it as default.
                    pass

                # Set stats rewards directly (can't use curriculum buckets for dict keys with dots)
                # Reward for gaining resources (not just having them) to avoid rewarding initial inventory
                if not mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats = {}
                # Set small rewards for resource collection - reward agents for gaining resources
                # These are fixed values since curriculum buckets can't handle dict keys with dots
                mission_env.game.agent.rewards.stats.setdefault("carbon.gained", 0.01)
                mission_env.game.agent.rewards.stats.setdefault("oxygen.gained", 0.01)
                mission_env.game.agent.rewards.stats.setdefault("germanium.gained", 0.01)
                mission_env.game.agent.rewards.stats.setdefault("silicon.gained", 0.01)

                mission_tasks = cc.bucketed(mission_env)

                # Add curriculum buckets for learning progress
                mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])
                # Note: stats rewards with dots in key names can't be set via curriculum buckets
                # We use inventory rewards for heart which get converted to stats internally
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])

                all_mission_tasks.append(mission_tasks)
            except Exception as e:
                # Log the error but continue - some variant combinations may be incompatible
                import logging

                logging.warning(f"Failed to create mission-variant combination {mission_name}+{variant_name}: {e}")
                continue

    if not all_mission_tasks:
        raise ValueError(f"No valid mission-variant combinations found for missions: {base_missions}")

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
    base_missions: list[str],
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    exclude_variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> tools.TrainTool:
    """Create a training tool for CoGs vs Clips with variants curriculum.

    Args:
        base_missions: List of mission names to include (required)
        num_cogs: Number of agents per mission
        curriculum: Optional curriculum configuration (defaults to variants curriculum)
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        exclude_variants: Optional list of variant names to exclude from the curriculum
        eval_variants: Optional mission variants to apply during evaluation
        eval_difficulty: Difficulty variant for evaluation

    Returns:
        A TrainTool configured with the variants curriculum
    """
    resolved_curriculum = curriculum or make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        exclude_variants=exclude_variants,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    resolved_eval_variants = cogs_v_clips._resolve_eval_variants(None, eval_variants)
    eval_suite = cogs_v_clips.make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
    )

    return tools.TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: list[str] | str,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> tools.EvaluateTool:
    """Evaluate policies on CoGs vs Clips missions."""
    return tools.EvaluateTool(
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
    mission: str = "training_facility.harvest",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> tools.PlayTool:
    """Play a single mission with a policy."""
    env = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )
    sim = SimulationConfig(suite="cogs_vs_clips", name=f"{mission}_{num_cogs}cogs", env=env)
    return tools.PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    base_missions: list[str],
    run_name: Optional[str] = None,
    num_cogs: int = 4,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    exclude_variants: Optional[list[str]] = None,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        base_missions: List of mission names to include (required)
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per mission (default: 4).
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600).
        skip_git_check: Whether to skip git check (default: True).
        exclude_variants: Optional list of variant names to exclude from the curriculum.
        additional_args: Additional arguments to pass to the training command.
    """
    if run_name is None:
        run_name = f"variants_curriculum_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.variants_curriculum.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        f"--heartbeat-timeout={heartbeat_timeout}",
    ]

    # Pass base_missions as a JSON list argument for hydra/omegaconf
    # Format: base_missions=["mission1","mission2","mission3"]
    missions_str = json.dumps(base_missions)
    cmd.append(f"base_missions={missions_str}")

    if skip_git_check:
        cmd.append("--skip-git-check")

    if exclude_variants:
        exclude_str = json.dumps(exclude_variants)
        cmd.append(f"exclude_variants={exclude_str}")

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
    "get_all_variant_names",
    "DIAGNOSTIC_MISSIONS",
]
