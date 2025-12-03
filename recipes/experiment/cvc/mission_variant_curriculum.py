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
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.missions import (
    HarvestMission,
    RepairMission,
    VibeCheckMission,
)
from cogames.cogs_vs_clips.variants import VARIANTS
from metta.agent.policies.puffer import PufferPolicyConfig
from metta.agent.policies.trxl import TRXLConfig
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


def resolve_missions(
    missions: Optional[list[str] | str] = None,
) -> list[str]:
    """Resolve mission set names or individual mission names to a list of mission names.

    Args:
        missions: Can be:
            - None: Returns FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A comma-separated string of mission names or set names
            - A list of mission names or set names

    Returns:
        List of mission name strings

    Examples:
        >>> resolve_missions("eval_missions")
        ['go_together', 'oxygen_bottleneck', ...]
        >>> resolve_missions(["eval_missions", "diagnostic_missions"])
        ['go_together', ..., 'diagnostic_assemble_seeded_near', ...]
        >>> resolve_missions("extractor_hub_30,extractor_hub_50")
        ['extractor_hub_30', 'extractor_hub_50']
    """
    # Handle None - default to all missions
    if missions is None:
        return list(FULL_CURRICULUM_MISSIONS)

    # Handle comma-separated string input (for shell compatibility)
    if isinstance(missions, str):
        missions = [m.strip() for m in missions.split(",") if m.strip()]

    if not missions:
        raise ValueError("missions must be non-empty")

    # Mission set name -> mission name list mapping
    MISSION_SETS: dict[str, tuple[str, ...]] = {
        "eval_missions": MISSIONS,
        "diagnostic_missions": DIAGNOSTIC_MISSIONS,
        "training_facility_missions": TRAINING_FACILITY_MISSIONS,
        "all": FULL_CURRICULUM_MISSIONS,
    }

    resolved: list[str] = []
    for item in missions:
        item = item.strip()
        # Check if it's a mission set name
        if item in MISSION_SETS:
            resolved.extend(MISSION_SETS[item])
        else:
            # Assume it's an individual mission name
            resolved.append(item)

    # Remove duplicates while preserving order
    seen = set()
    unique_resolved = []
    for mission in resolved:
        if mission not in seen:
            seen.add(mission)
            unique_resolved.append(mission)

    return unique_resolved


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
    variants: Optional[Sequence[str] | str] = None,
    stats_max_cap: float = 0.5,
) -> CurriculumConfig:
    """Create a mission-variant curriculum for CoGs vs Clips training with learning progress.

    Args:
        base_missions: Mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A comma-separated string of mission names or set names (e.g., "eval_missions,diagnostic_missions")
            - A list of mission names or set names (e.g., ["eval_missions", "extractor_hub_30"])
        num_cogs: Number of agents per mission
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        variants: Mission variants to apply. Can be:
            - None: No variants applied (base missions only)
            - "all": All variants applied (creates separate tasks for each mission-variant combination)
            - A list or comma-separated string of specific variant names
            If provided, creates separate tasks for each mission-variant combination.
        stats_max_cap: Maximum reward cap for resource stats (default: 0.5)

    Returns:
        A CurriculumConfig with learning progress algorithm
    """
    # Resolve mission sets to actual mission names
    base_missions = resolve_missions(base_missions)

    # Determine which variants to use
    if variants is None:
        # No variants at all - just base missions
        variant_names = []
    elif isinstance(variants, str) and variants.strip().lower() == "all":
        # Special case: "all" as a string means use all variants
        variant_names = get_all_variant_names()
    elif isinstance(variants, list) and len(variants) == 1 and variants[0].strip().lower() == "all":
        # Special case: ["all"] means use all variants
        variant_names = get_all_variant_names()
    else:
        # Handle comma-separated string for variants or use the specified list
        if isinstance(variants, str):
            variant_names = [v.strip() for v in variants.split(",") if v.strip()]
        else:
            # Use the specified variants
            variant_names = list(variants)

    all_mission_tasks = []

    if variant_names:
        # Create separate tasks for each mission-variant combination
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
        # No variants: just create tasks for base missions
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
                        variants=None,  # No variants
                    )
                except ValueError:
                    # Skip missions that don't exist
                    continue
            else:
                # Use the mission template directly (works for both eval and diagnostic missions)
                mission = cogs_v_clips._prepare_mission(
                    mission_template,
                    num_cogs=num_cogs,
                    variant_names=[],  # No variants
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
    variants: Optional[Sequence[str] | str] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    arch_type: str = "default",
) -> TrainTool:
    """Create a training tool for CoGs vs Clips with mission-variant curriculum.

    Args:
        base_missions: Mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A comma-separated string of mission names or set names
            - A list of mission names or set names
        num_cogs: Number of agents per mission
        curriculum: Optional curriculum configuration (defaults to mission-variant curriculum)
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        variants: Mission variants to apply. Can be:
            - None: No variants applied (base missions only)
            - "all": All variants applied (creates separate tasks for each mission-variant combination)
            - A list or comma-separated string of specific variant names
        eval_variants: Optional mission variants to apply during evaluation
        eval_difficulty: Difficulty variant for evaluation

    Returns:
        A TrainTool configured with the mission-variant curriculum
    """
    # Determine stats_max_cap based on whether variants are used
    # If variants="all" or variants is a list, use 0.5 (curriculum mode)
    # If variants=None, use 1.0 (full curriculum mode)
    if variants is None:
        stats_max_cap = 1.0
    else:
        # Check if variants is "all"
        if isinstance(variants, str) and variants.lower() == "all":
            stats_max_cap = 0.5
        elif isinstance(variants, list) and len(variants) == 1 and variants[0].lower() == "all":
            stats_max_cap = 0.5
        elif variants:  # Non-empty list of specific variants
            stats_max_cap = 0.5
        else:
            stats_max_cap = 1.0

    resolved_curriculum = curriculum or make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        stats_max_cap=stats_max_cap,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # For evaluation, convert "all" to None (evaluation doesn't use "all variants")
    # Only use specific variants if provided, otherwise use eval_variants or None
    eval_train_variants = None
    is_all_variants = variants == "all" or (
        isinstance(variants, list) and len(variants) == 1 and variants[0].lower() == "all"
    )
    if variants and not is_all_variants:
        eval_train_variants = variants
    resolved_eval_variants = cogs_v_clips._resolve_eval_variants(eval_train_variants, eval_variants)
    eval_suite = cogs_v_clips.make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
    )

    kwargs = {
        "trainer": trainer_cfg,
        "training_env": TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        "evaluator": evaluator_cfg,
    }

    if arch_type == "trxl":
        kwargs["policy_architecture"] = TRXLConfig()
    elif arch_type == "lstm":
        kwargs["policy_architecture"] = PufferPolicyConfig()
    elif arch_type != "default":
        raise ValueError(f"Unknown arch_type={arch_type!r} (expected 'default', 'trxl', or 'lstm')")

    return TrainTool(**kwargs)


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
    mission: str = "easy_hearts",
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
    variants: Optional[list[str] | str] = None,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        base_missions: Optional mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A list of mission names or set names
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per mission (default: 4).
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600).
        skip_git_check: Whether to skip git check (default: True).
        variants: Mission variants to apply. Can be:
            - None: No variants applied (base missions only)
            - "all": All variants applied (creates separate tasks for each mission-variant combination)
            - A list of specific variant names
        additional_args: Additional arguments to pass to the training command.
    """
    if run_name is None:
        if variants == "all" or (isinstance(variants, list) and len(variants) == 1 and variants[0].lower() == "all"):
            mode_str = "variants"
        elif variants:
            mode_str = "variants"
        else:
            mode_str = "full"
        run_name = f"mission_variant_curriculum_{mode_str}_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.mission_variant_curriculum.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        f"--heartbeat-timeout-seconds={heartbeat_timeout}",
    ]

    if base_missions:
        # Pass base_missions as comma-separated string (shell-safe format)
        missions_str = ",".join(base_missions)
        cmd.append(f"base_missions={missions_str}")

    if skip_git_check:
        cmd.append("--skip-git-check")

    if variants:
        if isinstance(variants, list):
            variants_str = ",".join(variants)
        else:
            variants_str = str(variants)
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
