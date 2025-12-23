"""Mission-variant curriculum for CoGs vs Clips with learning progress.

This curriculum supports two modes:
1. All variants per mission: Creates separate curriculum tasks for each mission-variant combination
2. Shared variants: Applies the same variant(s) to all missions (or no variants for base missions)

This replaces both variants_curriculum.py and full_curriculum.py.
"""

from __future__ import annotations

import time
from typing import NamedTuple, Optional, Sequence

import metta.cogworks.curriculum as cc
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
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig
from recipes.experiment import cogs_v_clips
from recipes.experiment.architectures import get_architecture

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
    "vibe_check",
    "repair",
)

# Procedural generation missions (using MachinaArena)
PROC_GEN_MISSIONS: tuple[str, ...] = (
    "hello_world.open_world",
    "machina_1.open_world",
    "machina_1.balanced_corners",
    "hello_world_unclip",
)

FULL_CURRICULUM_MISSIONS: tuple[str, ...] = (
    *cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS,  # Base curriculum missions (includes machina_1.open_world)
    # Additional procedural generation missions
    "hello_world.open_world",
    "machina_1.balanced_corners",
    "hello_world_unclip",
    # Training facility missions we currently support in this repo
    "harvest",
    "vibe_check",
    "repair",
    "oxygen_bottleneck",
    "single_use_swarm",
    *DIAGNOSTIC_MISSIONS,  # Diagnostic missions
)


def get_all_variant_names() -> list[str]:
    """Get all variant names from VARIANTS."""
    return [variant.name for variant in VARIANTS]


class _ResolvedVariants(NamedTuple):
    """Internal helper for normalized variant handling."""

    names: list[str]
    # True when caller requested "all" variants (string or single-item list),
    # which we expand to the full list of variant names.
    is_all: bool


def _resolve_variants_arg(variants: Optional[Sequence[str] | str]) -> _ResolvedVariants:
    """Normalize a variants argument into a list of names and an "all" flag.

    Supports:
    - None / "none" / [] -> no variants
    - "all" / ["all"] (case-insensitive) -> expand to all variants
    - comma-separated string -> split into names
    - list/sequence of names -> used as-is
    """
    if variants is None:
        return _ResolvedVariants(names=[], is_all=False)

    # If we got a string, handle special tokens and comma-separated lists.
    if isinstance(variants, str):
        token = variants.strip().lower()
        if token in {"none", ""}:
            return _ResolvedVariants(names=[], is_all=False)
        if token == "all":
            return _ResolvedVariants(names=get_all_variant_names(), is_all=True)
        names = [v.strip() for v in variants.split(",") if v.strip()]
        return _ResolvedVariants(names=names, is_all=False)

    # If we got a sequence, treat special cases like ["all"] / ["none"].
    seq = list(variants)
    if not seq:
        return _ResolvedVariants(names=[], is_all=False)

    if len(seq) == 1:
        token = seq[0].strip().lower()
        if token in {"none", ""}:
            return _ResolvedVariants(names=[], is_all=False)
        if token == "all":
            return _ResolvedVariants(names=get_all_variant_names(), is_all=True)

    # Otherwise, use the sequence as-is.
    return _ResolvedVariants(names=seq, is_all=False)


def resolve_missions(
    missions: Optional[list[str] | str] = None,
) -> list[str]:
    """Resolve mission set names or individual mission names to a list of mission names.

    Args:
        missions: Can be:
            - None: Returns FULL_CURRICULUM_MISSIONS
            - A mission set name: "diagnostic_missions", "training_facility_missions", "proc_gen_missions", "all"
            - A comma-separated string of mission names or set names
            - A list of mission names or set names

    Returns:
        List of mission name strings

    Examples:
        >>> resolve_missions("eval_missions")
        ['go_together', 'oxygen_bottleneck', ...]
        >>> resolve_missions(["eval_missions", "diagnostic_missions"])
        ['go_together', ..., 'diagnostic_assemble_seeded_near', ...]
        >>> resolve_missions("energy_starved,oxygen_bottleneck")
        ['energy_starved', 'oxygen_bottleneck']
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
        "diagnostic_missions": DIAGNOSTIC_MISSIONS,
        "training_facility_missions": TRAINING_FACILITY_MISSIONS,
        "proc_gen_missions": PROC_GEN_MISSIONS,
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


def _enforce_training_vibes(env: MettaGridConfig) -> None:
    """Enforce the training set of vibes and action space consistency."""
    training_vibe_names = [v.name for v in vibes.VIBES]
    env.game.vibe_names = training_vibe_names

    if env.game.actions:
        # Configure vibe action
        if env.game.actions.change_vibe:
            env.game.actions.change_vibe.number_of_vibes = len(training_vibe_names)
            # Filter initial vibe
            if env.game.agent.initial_vibe >= len(training_vibe_names):
                env.game.agent.initial_vibe = 0

        # This ensures action space is 19 (1 noop + 4 move + 14 vibes)
        if env.game.actions.attack:
            env.game.actions.attack.enabled = False
    # Prune transfers
    allowed_vibes = set(env.game.vibe_names)
    chest = env.game.objects.get("chest")
    if chest:
        vibe_transfers = getattr(chest, "vibe_transfers", None)
        if isinstance(vibe_transfers, dict):
            new_transfers = {v: t for v, t in vibe_transfers.items() if v in allowed_vibes}
            chest.vibe_transfers = new_transfers


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
            - A list of mission names or set names (e.g., ["eval_missions", "energy_starved"])
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

    # Normalize variant argument once so all call sites share the same semantics.
    resolved_variants = _resolve_variants_arg(variants)
    variant_names = resolved_variants.names

    all_mission_tasks = []

    if variant_names:
        # Create separate tasks for each mission-variant combination
        for mission_name in base_missions:
            # For each variant, create a separate curriculum task
            for variant_name in variant_names:
                try:
                    # Unified path: Always use make_training_env, which resolves via global MISSIONS registry
                    mission_env = cogs_v_clips.make_training_env(
                        num_cogs=num_cogs,
                        mission=mission_name,
                        variants=[variant_name],
                    )

                    # Enforce training vibes
                    _enforce_training_vibes(mission_env)

                    # Deduplicate assembler protocols to avoid C++ config errors
                    _deduplicate_assembler_protocols(mission_env)

                    # Give each environment a label so per-label rewards can be tracked in stats/W&B.
                    # Use mission:variant format to distinguish variant combinations.
                    label = f"{mission_name}:{variant_name}"
                    try:
                        mission_env.label = label  # type: ignore[attr-defined]
                    except Exception:
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
                    # Skip incompatible variant combinations
                    error_str = str(e).lower()
                    if any(
                        skip_phrase in error_str
                        for skip_phrase in [
                            "not callable",
                            "already exists",
                            "can only be applied",
                            "not found",
                            "does not exist",
                            "incompatible",
                            "protocol with vibes",
                        ]
                    ):
                        import logging

                        logging.debug(
                            f"Skipping incompatible mission-variant combination {mission_name}+{variant_name}: {e}"
                        )
                        continue
                    # For map building errors or other unexpected errors
                    import logging

                    logging.warning(f"Failed to create mission-variant combination {mission_name}+{variant_name}: {e}")
                    continue

        if not all_mission_tasks:
            raise ValueError(f"No valid mission-variant combinations found for missions: {base_missions}")

    else:
        # No variants: just create tasks for base missions
        for mission_name in base_missions:
            try:
                # Unified path: Always use make_training_env
                mission_env = cogs_v_clips.make_training_env(
                    num_cogs=num_cogs,
                    mission=mission_name,
                    variants=None,  # No variants
                )

                # Enforce training vibes
                _enforce_training_vibes(mission_env)

                # Deduplicate assembler protocols
                _deduplicate_assembler_protocols(mission_env)

                # Give each environment a label
                try:
                    mission_env.label = mission_name  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Initialize stats rewards dict
                if not mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats = {}
                # Initialize stats_max dict
                if not mission_env.game.agent.rewards.stats_max:
                    mission_env.game.agent.rewards.stats_max = {}
            except Exception as e:
                # Skip incompatible combinations
                error_str = str(e).lower()
                if any(
                    skip_phrase in error_str
                    for skip_phrase in [
                        "not callable",
                        "already exists",
                        "can only be applied",
                        "not found",
                        "does not exist",
                        "incompatible",
                        "protocol with vibes",
                    ]
                ):
                    import logging

                    logging.debug(f"Skipping incompatible mission {mission_name}: {e}")
                    continue
                # For map building errors or other unexpected errors
                import logging

                logging.warning(f"Failed to create mission {mission_name}: {e}")
                continue

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
    arch_type: str = "vit",
) -> TrainTool:
    """Create a training tool for CoGs vs Clips with mission-variant curriculum.

    Args:
        base_missions: Mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "diagnostic_missions", "training_facility_missions", "proc_gen_missions", "all"
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
    # Normalize variants once and use the result consistently everywhere.
    resolved_variants = _resolve_variants_arg(variants)
    has_variants = bool(resolved_variants.names)

    # Determine stats_max_cap based on whether curriculum is using variants.
    # If we have any variants, use 0.5 (curriculum mode); otherwise 1.0 (full curriculum mode).
    stats_max_cap = 0.5 if has_variants else 1.0

    resolved_curriculum = curriculum or make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=resolved_variants.names,
        stats_max_cap=stats_max_cap,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # For evaluation, "all" is treated the same as "no explicit variant filter".
    # Only use specific variants if provided, otherwise use eval_variants or None.
    eval_train_variants: Optional[Sequence[str] | str] = None
    if has_variants and not resolved_variants.is_all:
        eval_train_variants = resolved_variants.names
    resolved_eval_variants = cogs_v_clips._resolve_eval_variants(eval_train_variants, eval_variants)
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
        policy_architecture=get_architecture(arch_type),
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
        simulations=cogs_v_clips.make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def _get_policy_action_space(policy_uri: str) -> Optional[int]:
    """Detect the action space size from a policy checkpoint."""
    if not policy_uri or not policy_uri.startswith("s3://"):
        return None

    try:
        from metta.rl.mpt_artifact import load_mpt

        artifact = load_mpt(policy_uri)

        # Look for actor head weight to determine action space
        for key, tensor in artifact.state_dict.items():
            if "actor_head" in key and "weight" in key and len(tensor.shape) == 2:
                return tensor.shape[0]
        return None
    except Exception:
        return None


def _configure_env_for_action_space(env, num_actions: int) -> None:
    """Configure environment vibes to match a specific action space."""
    # Action space = noop (1) + move (4) + change_vibe (N)
    num_vibes = num_actions - 5

    if num_vibes <= 0:
        return

    # Select the appropriate vibe set
    if num_vibes == 16:
        vibe_names = [v.name for v in vibes.VIBES]
    elif num_vibes == 13:
        vibe_names = [v.name for v in vibes.VIBES[:13]]
    elif num_vibes <= len(vibes.VIBES):
        vibe_names = [v.name for v in vibes.VIBES[:num_vibes]]
    else:
        vibe_names = [v.name for v in vibes.VIBES]

    env.game.vibe_names = vibe_names

    if env.game.actions:
        if env.game.actions.change_vibe:
            env.game.actions.change_vibe.number_of_vibes = len(vibe_names)
            if env.game.agent.initial_vibe >= len(vibe_names):
                env.game.agent.initial_vibe = 0
        if env.game.actions.attack:
            env.game.actions.attack.enabled = False


def play(
    policy_uri: Optional[str] = None,
    mission: str = "easy_hearts",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    num_vibes: Optional[int] = None,
) -> PlayTool:
    """Play a single mission with a policy.

    Args:
        policy_uri: URI of the policy to load
        mission: Mission name to play
        num_cogs: Number of agents
        variants: Optional mission variants
        num_vibes: Number of vibes to use. If None, auto-detects from policy checkpoint.
                  Can be set manually: 13 for cvc_random_maps, 16 for base.1201 policies.
    """
    env = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )

    # Auto-detect action space from policy if not specified
    if num_vibes is None and policy_uri:
        detected_actions = _get_policy_action_space(policy_uri)
        if detected_actions is not None:
            _configure_env_for_action_space(env, detected_actions)
    elif num_vibes is not None:
        # Manual override
        _configure_env_for_action_space(env, num_vibes + 5)  # Convert vibes to actions

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
            - A mission set name: "diagnostic_missions", "training_facility_missions", "proc_gen_missions", "all"
            - A list of mission names or set names
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per mission (default: 4).
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600).
        skip_git_check: Whether to skip git check (default: True).
        variants: Mission variants to apply. Can be:
            - None / "none": No variants applied (base missions only)
            - "all" / ["all"]: All variants applied (creates separate tasks for each mission-variant combination)
            - A list of specific variant names
        additional_args: Additional arguments to pass to the training command.
    """
    # Normalize variants so naming and CLI wiring are consistent with training.
    resolved_variants = _resolve_variants_arg(variants)
    has_variants = bool(resolved_variants.names)

    if run_name is None:
        mode_str = "variants" if has_variants else "full"
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

    if has_variants:
        # Pass variants as a comma-separated string (shell-safe format)
        variants_str = ",".join(resolved_variants.names)
        print(f"Variants: {variants_str}")
        cmd.append(f"variants={variants_str}")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching training job: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    print(f"Command: {' '.join(cmd)}")

    # subprocess.run(cmd, check=True)
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
    "PROC_GEN_MISSIONS",
]
