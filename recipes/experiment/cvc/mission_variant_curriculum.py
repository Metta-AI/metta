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


# Proven variants from successful variant-focused curricula
# Based on analysis: go_together and collect_resources variants achieved 100% success
PROVEN_VARIANTS: tuple[str, ...] = (
    # Clipping variants
    "clipped_oxygen",
    "clipped_silicon",
    "clipped_carbon",
    "clipped_germanium",
    "clipping_chaos",
    # Terrain variants
    "rough_terrain",
    "mined_out",
    "dark_side",
    # Tuning variants
    "extractor_heart_tune",
    "chest_heart_tune",
    "inventory_heart_tune",
    # Constraint variants
    "cog_tools_only",
    "single_tool_unclip",
    "cyclical_unclip",
    # Environmental variants
    "energy_crisis",
    "solar_flare",
    "compass",
    "energized",
    # Heart protocol variants
    "heart_chorus",
    "lonely_heart",
    "tiny_heart_protocols",
    "vibe_check_min_2",
    # Other proven variants
    "small_50",
    "super_charged",
    "trader",
    "pack_rat",
    "standard",
    "clip_hub_stations",
    "clip_period_on",
    "resource_bottleneck",
)

# Mission difficulty tiers (cumulative)
# Tier 1: Easy/Proven missions that succeeded with all variants
# Tier 2: Medium difficulty, includes Tier 1
# Tier 3: Hard missions that need diversity, includes Tier 1+2
MISSION_TIERS: dict[str, list[str]] = {
    "tier1_easy": [
        "go_together",
        "collect_resources_classic",
        "collect_resources_spread",
    ],
    "tier2_medium": [
        # Tier 1 included
        "go_together",
        "collect_resources_classic",
        "collect_resources_spread",
        # Tier 2 additions
        "extractor_hub_30",
        "extractor_hub_50",
        "extractor_hub_70",
        "divide_and_conquer",
        "harvest",
        "repair",
        "vibe_check",
    ],
    "tier3_hard": [
        # Tier 1+2 included
        "go_together",
        "collect_resources_classic",
        "collect_resources_spread",
        "extractor_hub_30",
        "extractor_hub_50",
        "extractor_hub_70",
        "divide_and_conquer",
        "harvest",
        "repair",
        "vibe_check",
        # Tier 3 additions
        "oxygen_bottleneck",
        "energy_starved",
        "collect_far",
    ],
}

# Full curriculum without diagnostics (for experiments testing impact)
FULL_CURRICULUM_MISSIONS_NO_DIAGNOSTICS: tuple[str, ...] = (
    *cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS,
    *MISSIONS,
    "harvest",
    "assemble",
    "vibe_check",
    "repair",
    # Diagnostics excluded - all mastered, may be too easy
)


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
    MISSION_SETS: dict[str, tuple[str, ...] | list[str]] = {
        "eval_missions": MISSIONS,
        "diagnostic_missions": DIAGNOSTIC_MISSIONS,
        "training_facility_missions": TRAINING_FACILITY_MISSIONS,
        "all": FULL_CURRICULUM_MISSIONS,
        "all_no_diagnostics": FULL_CURRICULUM_MISSIONS_NO_DIAGNOSTICS,
        # Tier-based mission sets
        "tier1": MISSION_TIERS["tier1_easy"],
        "tier2": MISSION_TIERS["tier2_medium"],
        "tier3": MISSION_TIERS["tier3_hard"],
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
    exclude_variants: Optional[Sequence[str] | str] = None,
    stats_max_cap: float = 0.5,
    use_proven_variants_only: bool = False,
    progressive_deposit_rewards: bool = True,
    adjusted_inventory_rewards: bool = True,
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
        variants: Optional mission variants to apply. If None, no variants are applied.
            If provided, creates separate tasks for each mission-variant combination.
            Can be a list or comma-separated string.
        exclude_variants: Optional list of variant names to exclude, or comma-separated string.
            Only used when variants=None (to get all variants except excluded ones).
        stats_max_cap: Maximum reward cap for resource stats (default: 0.5)
        use_proven_variants_only: If True, only use variants from PROVEN_VARIANTS list.
            Based on analysis, these variants succeeded in go_together/collect_resources runs.
        progressive_deposit_rewards: If True, use progressive deposit reward buckets [1.5, 2.0, 2.5, 3.0].
            If False, use simpler buckets [1.0, 2.0, 3.0].
        adjusted_inventory_rewards: If True, use adjusted inventory rewards [0.1, 0.2, 0.3, 0.5] for better gradient.
            If False, use original [0.1, 0.333, 0.5, 1.0].

    Returns:
        A CurriculumConfig with learning progress algorithm
    """
    # Resolve mission sets to actual mission names
    base_missions = resolve_missions(base_missions)

    # Handle comma-separated string for variants
    if isinstance(variants, str):
        variants = [v.strip() for v in variants.split(",") if v.strip()]

    # Handle comma-separated string for exclude_variants
    if isinstance(exclude_variants, str):
        exclude_variants = [v.strip() for v in exclude_variants.split(",") if v.strip()]

    # Determine which variants to use
    if variants is None:
        # If no variants specified, check if we should use all variants (minus excluded)
        if exclude_variants is not None:
            # Get all variants except excluded ones
            all_variant_names = get_all_variant_names()
            exclude_set = set(exclude_variants)
            variant_names = [v for v in all_variant_names if v not in exclude_set]
        else:
            # No variants at all - just base missions
            variant_names = []
    else:
        # Use the specified variants
        variant_names = list(variants)

    # Filter to proven variants only if requested
    if use_proven_variants_only and variant_names:
        proven_set = set(PROVEN_VARIANTS)
        variant_names = [v for v in variant_names if v in proven_set]
        if not variant_names:
            import logging

            logging.warning("use_proven_variants_only=True but no proven variants found. Falling back to all variants.")
            variant_names = list(PROVEN_VARIANTS)

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
                    if adjusted_inventory_rewards:
                        mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.2, 0.3, 0.5])
                    else:
                        mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])
                    # CRITICAL: Add chest deposit reward buckets to encourage depositing hearts
                    # Make deposit rewards 2-3x inventory heart rewards at same bucket level
                    if progressive_deposit_rewards:
                        mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.5, 2.0, 2.5, 3.0])
                    else:
                        mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.0, 2.0, 3.0])

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
            if adjusted_inventory_rewards:
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.2, 0.3, 0.5])
            else:
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])
            # CRITICAL: Add chest deposit reward buckets to encourage depositing hearts
            # Make deposit rewards 2-3x inventory heart rewards at same bucket level
            if progressive_deposit_rewards:
                mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.5, 2.0, 2.5, 3.0])
            else:
                mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.0, 2.0, 3.0])

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


def make_eval_suite_from_curriculum(
    base_missions: Optional[list[str] | str] = None,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    exclude_variants: Optional[Sequence[str] | str] = None,
) -> list[SimulationConfig]:
    """Create an eval suite that matches the mission:variant combinations from training.

    When all_variants_per_mission=True, this creates eval environments for each
    mission:variant combination that was trained on, ensuring evaluation matches training.

    Args:
        base_missions: Mission names (same as used in training)
        num_cogs: Number of agents per mission
        variants: Variant names (if None and exclude_variants is None, uses all variants)
        exclude_variants: Variant names to exclude

    Returns:
        A list of SimulationConfig objects matching the training mission:variant combinations
    """
    # Resolve mission sets to actual mission names (same logic as make_curriculum)
    base_missions = resolve_missions(base_missions)

    # Handle comma-separated string for variants
    if isinstance(variants, str):
        variants = [v.strip() for v in variants.split(",") if v.strip()]

    # Handle comma-separated string for exclude_variants
    if isinstance(exclude_variants, str):
        exclude_variants = [v.strip() for v in exclude_variants.split(",") if v.strip()]

    # Determine which variants to use (same logic as make_curriculum)
    if variants is None:
        if exclude_variants is not None:
            all_variant_names = get_all_variant_names()
            exclude_set = set(exclude_variants)
            variant_names = [v for v in all_variant_names if v not in exclude_set]
        else:
            variant_names = []
    else:
        variant_names = list(variants)

    simulations: list[SimulationConfig] = []

    if variant_names:
        # Create eval environments for each mission:variant combination
        for mission_name in base_missions:
            mission_template = _MISSION_BY_NAME.get(mission_name)

            # Check if this is a diagnostic mission (class-based)
            if mission_template is None and mission_name in DIAGNOSTIC_MISSIONS:
                for mission_cls in DIAGNOSTIC_EVALS:
                    temp_mission = mission_cls()  # type: ignore[call-arg]
                    if temp_mission.name == mission_name:
                        mission_template = temp_mission
                        break

            for variant_name in variant_names:
                try:
                    if mission_template is None:
                        # Fall back to make_training_env for standard missions
                        try:
                            env_cfg = cogs_v_clips.make_training_env(
                                num_cogs=num_cogs,
                                mission=mission_name,
                                variants=[variant_name],
                            )
                        except ValueError:
                            continue
                    else:
                        # Use the mission template directly
                        mission = cogs_v_clips._prepare_mission(
                            mission_template,
                            num_cogs=num_cogs,
                            variant_names=[variant_name],
                        )
                        env_cfg = mission.make_env()

                    # Create simulation config matching the training label format
                    sim = SimulationConfig(
                        suite="cogs_vs_clips",
                        name=f"{mission_name}:{variant_name}_{num_cogs}cogs",
                        env=env_cfg,
                    )
                    simulations.append(sim)
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to create eval env for {mission_name}:{variant_name}: {e}")
                    continue
    else:
        # No variants: create eval environments for base missions only
        for mission_name in base_missions:
            mission_template = _MISSION_BY_NAME.get(mission_name)

            if mission_template is None and mission_name in DIAGNOSTIC_MISSIONS:
                for mission_cls in DIAGNOSTIC_EVALS:
                    temp_mission = mission_cls()  # type: ignore[call-arg]
                    if temp_mission.name == mission_name:
                        mission_template = temp_mission
                        break

            try:
                if mission_template is None:
                    env_cfg = cogs_v_clips.make_training_env(
                        num_cogs=num_cogs,
                        mission=mission_name,
                        variants=None,
                    )
                else:
                    mission = cogs_v_clips._prepare_mission(
                        mission_template,
                        num_cogs=num_cogs,
                        variant_names=[],
                    )
                    env_cfg = mission.make_env()

                sim = SimulationConfig(
                    suite="cogs_vs_clips",
                    name=f"{mission_name}_{num_cogs}cogs",
                    env=env_cfg,
                )
                simulations.append(sim)
            except Exception as e:
                import logging

                logging.warning(f"Failed to create eval env for {mission_name}: {e}")
                continue

    return simulations


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
    eval_match_training: bool = True,
    use_proven_variants_only: bool = False,
    progressive_deposit_rewards: bool = True,
    adjusted_inventory_rewards: bool = True,
) -> TrainTool:
    """Create a training tool for CoGs vs Clips with mission-variant curriculum.

    Args:
        base_missions: Mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A comma-separated string of mission names or set names
            - A list of mission names or set names
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
        eval_match_training: If True and all_variants_per_mission=True, create eval environments
            that match the training mission:variant combinations. If False, use standard eval suite.

    Returns:
        A TrainTool configured with the mission-variant curriculum
    """
    # When all_variants_per_mission=True, we want all variants (unless exclude_variants is specified)
    # Pass exclude_variants=[] to indicate "get all variants" if not already specified
    resolved_exclude_variants = exclude_variants
    if all_variants_per_mission and exclude_variants is None:
        resolved_exclude_variants = []

    resolved_curriculum = curriculum or make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        exclude_variants=resolved_exclude_variants,
        stats_max_cap=0.5 if all_variants_per_mission else 1.0,
        use_proven_variants_only=use_proven_variants_only,
        progressive_deposit_rewards=progressive_deposit_rewards,
        adjusted_inventory_rewards=adjusted_inventory_rewards,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # If all_variants_per_mission=True and eval_match_training=True, create eval suite
    # that matches the training mission:variant combinations
    if all_variants_per_mission and eval_match_training:
        eval_suite = make_eval_suite_from_curriculum(
            base_missions=base_missions,
            num_cogs=num_cogs,
            variants=variants,
            exclude_variants=resolved_exclude_variants,
        )
    else:
        # Use standard eval suite (base missions with difficulty/variants)
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
        base_missions: Optional mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS (if all_variants_per_mission=False)
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A list of mission names or set names
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
    "FULL_CURRICULUM_MISSIONS_NO_DIAGNOSTICS",
    "DIAGNOSTIC_MISSIONS",
    "TRAINING_FACILITY_MISSIONS",
    "PROVEN_VARIANTS",
    "MISSION_TIERS",
]
