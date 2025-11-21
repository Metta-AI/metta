"""Dense curriculum for CoGs vs Clips with resource reduction levels.

This curriculum uses dense_training maps with varying resource density levels:
- Each map has 10 levels of resource density (1-10)
- Level 10 = full resources (100%)
- Level 1 = minimal resources (10%)
- Resources reduced: C (carbon), O (oxygen), G (germanium), S (silicon)
- Other elements (assemblers, chargers, chests) remain at full density
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.dense_envs import DENSE_TRAINING_MISSIONS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.map_utils.resource_reducer import reduce_map_resources
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
from mettagrid.config.mettagrid_config import MettaGridConfig


class ResourceReductionVariant(MissionVariant):
    """Apply resource reduction to a map based on difficulty level.

    This variant should be applied BEFORE the MapVariant in dense_training missions,
    or it should specify the map_name directly.

    Attributes:
        level: Resource density level (1-10, where 10 = 100% resources, 1 = 10% resources)
        map_name: Optional map name to load and reduce (if not provided, reduces current map)
        seed: Random seed for reproducible resource removal
    """

    name: str = "resource_reduction"
    level: int = 10
    map_name: str | None = None
    seed: int = 42

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        """Apply resource reduction to the map."""
        # Define resource levels based on the difficulty level
        # All resources get the same level for simplicity, but you can customize
        resource_levels = {
            "C": self.level,  # Carbon
            "O": self.level,  # Oxygen
            "G": self.level,  # Germanium
            "S": self.level,  # Silicon
            "&": 10,  # Assembler - always full
            "+": 10,  # Charger - always full
            "=": 10,  # Chest - always full
        }

        # Apply resource reduction by loading the map file directly
        # Pass the map_name (file path) to reduce_map_resources
        if self.map_name is None:
            raise ValueError("map_name must be provided for ResourceReductionVariant")
        reduced_map = reduce_map_resources(
            self.map_name,
            resource_levels=resource_levels,
            seed=self.seed,
        )

        # Update the environment with the reduced map
        env.game.map_builder = reduced_map


class MaxUsesVariant(MissionVariant):
    """Set max_uses for extractors (bucketed in curriculum)."""

    name: str = "max_uses"
    carbon_max_uses: int = 255
    oxygen_max_uses: int = 255
    silicon_max_uses: int = 255

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        """Set max_uses for extractors."""
        if "carbon_extractor" in env.game.objects:
            env.game.objects["carbon_extractor"].max_uses = self.carbon_max_uses
        if "oxygen_extractor" in env.game.objects:
            env.game.objects["oxygen_extractor"].max_uses = self.oxygen_max_uses
        if "silicon_extractor" in env.game.objects:
            env.game.objects["silicon_extractor"].max_uses = self.silicon_max_uses


# Map names for reference
MAP_NAMES = [
    "dense_training_4agents",
    "dense_training_4agentsbase",
    "dense_training_big",
    "dense_training_small",
]


def make_dense_curriculum(
    num_cogs: int = 4,
    resource_levels: list[int] | None = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
    use_all_maps: bool = True,
    maps_to_use: Optional[list[str] | str] = None,
    include_diagnostics: bool = False,
    max_uses_values: Optional[list[int] | str] = None,
) -> CurriculumConfig:
    """Create a dense curriculum with resource reduction levels.

    Args:
        num_cogs: Number of agents per environment (4 or 24)
        resource_levels: List of resource levels to include (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        variants: Optional additional mission variants to apply
        use_all_maps: If True, use all maps. If False, only use big/small maps (for 24-agent training)
        maps_to_use: Optional list of map names to include (overrides use_all_maps)
        include_diagnostics: If True, add diagnostic missions at full resources
        max_uses_values: List of max_uses values to bucket over for extractors

    Returns:
        A CurriculumConfig with learning progress algorithm across resource densities
    """
    if resource_levels is None:
        resource_levels = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Handle JSON string parsing for maps_to_use and max_uses_values
    parsed_maps_to_use = maps_to_use
    if isinstance(maps_to_use, str):
        try:
            parsed_maps_to_use = json.loads(maps_to_use)
        except (json.JSONDecodeError, TypeError):
            parsed_maps_to_use = [m.strip() for m in maps_to_use.split(",") if m.strip()]

    parsed_max_uses_values = max_uses_values
    if isinstance(max_uses_values, str):
        try:
            parsed_max_uses_values = json.loads(max_uses_values)
        except (json.JSONDecodeError, TypeError):
            parsed_max_uses_values = [int(v.strip()) for v in max_uses_values.split(",") if v.strip()]
    if not parsed_max_uses_values:
        parsed_max_uses_values = [1, 2, 3, 5, 10, 255]
    if parsed_max_uses_values is None:
        parsed_max_uses_values = [1, 2, 3, 5, 10, 255]

    all_mission_tasks = []

    # Filter missions based on maps_to_use or use_all_maps
    if parsed_maps_to_use is not None:
        missions_to_use = [m for m in DENSE_TRAINING_MISSIONS if m.name in parsed_maps_to_use]
    elif not use_all_maps:
        # Only use big/small maps for 24-agent training
        missions_to_use = [
            m for m in DENSE_TRAINING_MISSIONS if m.name in {"dense_training_big", "dense_training_small"}
        ]
    else:
        missions_to_use = DENSE_TRAINING_MISSIONS

    # Process each mission from dense_training
    for base_mission in missions_to_use:
        # Extract the map name from the base mission's MapVariant
        map_name = None
        other_variants = []
        for variant in base_mission.variants or []:
            if hasattr(variant, "map_name"):
                map_name = variant.map_name
            else:
                other_variants.append(variant)

        if map_name is None:
            raise ValueError(f"Could not find map_name in {base_mission.name} variants")

        # Create tasks for each resource level and max_uses combination
        for level in resource_levels:
            for max_uses in parsed_max_uses_values:
                # Create mission variants: ResourceReductionVariant + MaxUsesVariant
                mission_variants = [
                    ResourceReductionVariant(level=level, map_name=map_name, seed=42),
                    MaxUsesVariant(
                        carbon_max_uses=max_uses,
                        oxygen_max_uses=max_uses,
                        silicon_max_uses=max_uses,
                    ),
                    *other_variants,
                ]

                # Create mission with the reduction variant
                mission = Mission(
                    name=f"{base_mission.name}_level{level}_maxuses{max_uses}",
                    description=f"{base_mission.description} (Resource level {level}/10, max_uses={max_uses})",
                    site=base_mission.site,
                    variants=mission_variants,
                    num_cogs=num_cogs,
                )

                # Create environment
                mission_env = mission.make_env()

                # Initialize stats rewards dict if needed (for bucket overrides to work)
                if not mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats = {}
                # Initialize chest.heart.deposited key (treated as single key with dots, not nested)
                if "chest.heart.deposited" not in mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats["chest.heart.deposited"] = 0.0
                # Initialize stats_max dict if needed
                if not mission_env.game.agent.rewards.stats_max:
                    mission_env.game.agent.rewards.stats_max = {}

                # Initialize max_uses in env config before bucketing
                if "carbon_extractor" in mission_env.game.objects:
                    mission_env.game.objects["carbon_extractor"].max_uses = max_uses
                if "oxygen_extractor" in mission_env.game.objects:
                    mission_env.game.objects["oxygen_extractor"].max_uses = max_uses
                if "silicon_extractor" in mission_env.game.objects:
                    mission_env.game.objects["silicon_extractor"].max_uses = max_uses

                # Label the environment for tracking
                try:
                    mission_env.label = f"{base_mission.name}_rlvl{level}_maxuses{max_uses}"  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Create bucketed tasks
                mission_tasks = cc.bucketed(mission_env)

                # Scale max_steps by resource level
                if level <= 3:  # Sparse levels
                    max_steps_buckets = [1000, 1500, 2000, 2500, 3000]
                elif level <= 6:  # Medium levels
                    max_steps_buckets = [750, 1000, 1250, 1500, 2000]
                else:  # Dense levels (7-10)
                    max_steps_buckets = [500, 750, 1000, 1250, 1500]

                mission_tasks.add_bucket("game.max_steps", max_steps_buckets)

                # Add chest deposit reward buckets (CRITICAL)
                mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.5, 2.0, 2.5, 3.0])

                # Adjusted inventory heart rewards (from successful curricula)
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.2, 0.3, 0.5])

                # Reduced resource collection rewards
                mission_tasks.add_bucket("game.agent.rewards.inventory.carbon", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.oxygen", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.germanium", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.silicon", [0.0, 0.005, 0.01, 0.02])

                all_mission_tasks.append(mission_tasks)

    # Add diagnostic missions if requested
    if include_diagnostics:
        from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS

        for diagnostic_cls in DIAGNOSTIC_EVALS:
            diagnostic_mission = diagnostic_cls()  # type: ignore[call-arg]

            # Create tasks for each max_uses value (diagnostics at full resources, no reduction)
            for max_uses in parsed_max_uses_values:
                # Create mission with max_uses variant only (no resource reduction)
                mission_variants = [
                    MaxUsesVariant(
                        carbon_max_uses=max_uses,
                        oxygen_max_uses=max_uses,
                        silicon_max_uses=max_uses,
                    ),
                ]

                # Create a copy of the diagnostic mission with max_uses variant
                diagnostic_with_variant = Mission(
                    name=f"{diagnostic_mission.name}_maxuses{max_uses}",
                    description=f"{diagnostic_mission.description} (max_uses={max_uses})",
                    site=diagnostic_mission.site,
                    variants=mission_variants,
                    num_cogs=num_cogs,
                )

                # Create environment
                mission_env = diagnostic_with_variant.make_env()

                # Initialize stats rewards dict if needed
                if not mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats = {}
                if "chest.heart.deposited" not in mission_env.game.agent.rewards.stats:
                    mission_env.game.agent.rewards.stats["chest.heart.deposited"] = 0.0
                if not mission_env.game.agent.rewards.stats_max:
                    mission_env.game.agent.rewards.stats_max = {}

                # Initialize max_uses in env config
                if "carbon_extractor" in mission_env.game.objects:
                    mission_env.game.objects["carbon_extractor"].max_uses = max_uses
                if "oxygen_extractor" in mission_env.game.objects:
                    mission_env.game.objects["oxygen_extractor"].max_uses = max_uses
                if "silicon_extractor" in mission_env.game.objects:
                    mission_env.game.objects["silicon_extractor"].max_uses = max_uses

                # Label the environment
                try:
                    mission_env.label = f"{diagnostic_mission.name}_maxuses{max_uses}"  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Create bucketed tasks
                mission_tasks = cc.bucketed(mission_env)

                # Use dense level max_steps buckets for diagnostics
                mission_tasks.add_bucket("game.max_steps", [500, 750, 1000, 1250, 1500])

                # Add chest deposit reward buckets
                mission_tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited", [1.5, 2.0, 2.5, 3.0])

                # Adjusted inventory heart rewards
                mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.2, 0.3, 0.5])

                # Reduced resource collection rewards
                mission_tasks.add_bucket("game.agent.rewards.inventory.carbon", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.oxygen", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.germanium", [0.0, 0.005, 0.01, 0.02])
                mission_tasks.add_bucket("game.agent.rewards.inventory.silicon", [0.0, 0.005, 0.01, 0.02])

                all_mission_tasks.append(mission_tasks)

    # Merge all tasks
    merged_tasks = cc.merge(all_mission_tasks)

    # Configure learning progress algorithm
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=3000,  # Higher because we have more tasks
            max_slice_axes=4,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    # Adjust num_active_tasks based on number of tasks created
    num_tasks = len(all_mission_tasks)
    if num_tasks > 0:
        # Use a reasonable number of active tasks (roughly 50-75% of total tasks)
        num_active_tasks = min(2000, max(500, int(num_tasks * 0.6)))
    else:
        num_active_tasks = 2000

    return merged_tasks.to_curriculum(
        num_active_tasks=num_active_tasks,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    resource_levels: list[int] | None = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    use_all_maps: bool = True,
    maps_to_use: Optional[list[str] | str] = None,
    include_diagnostics: bool = False,
    max_uses_values: Optional[list[int] | str] = None,
) -> TrainTool:
    """Create a training tool for dense curriculum with resource reduction.

    Args:
        num_cogs: Number of agents per environment (4 or 24)
        curriculum: Optional curriculum configuration
        resource_levels: List of resource levels to include
        enable_detailed_slice_logging: Enable detailed logging
        variants: Optional mission variants for training
        eval_variants: Optional mission variants for evaluation
        use_all_maps: If True, use all 4 maps. If False, only big/small (for 24-agent training)
        maps_to_use: Optional list of map names to include (overrides use_all_maps)
        include_diagnostics: If True, add diagnostic missions at full resources
        max_uses_values: List of max_uses values to bucket over for extractors

    Returns:
        A TrainTool configured with the dense curriculum

    Examples:
        Train with 4 agents on all maps:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.train num_cogs=4

        Train with 24 agents on big/small maps only:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.train num_cogs=24 use_all_maps=false
    """
    # Handle JSON string parsing for maps_to_use and max_uses_values (from command line)
    parsed_maps_to_use = maps_to_use
    if isinstance(maps_to_use, str):
        try:
            parsed_maps_to_use = json.loads(maps_to_use)
        except (json.JSONDecodeError, TypeError):
            parsed_maps_to_use = [m.strip() for m in maps_to_use.split(",") if m.strip()]

    parsed_max_uses_values = max_uses_values
    if isinstance(max_uses_values, str):
        try:
            parsed_max_uses_values = json.loads(max_uses_values)
        except (json.JSONDecodeError, TypeError):
            parsed_max_uses_values = [int(v.strip()) for v in max_uses_values.split(",") if v.strip()]
    if parsed_max_uses_values is None:
        parsed_max_uses_values = [1, 2, 3, 5, 10, 255]

    resolved_curriculum = curriculum or make_dense_curriculum(
        num_cogs=num_cogs,
        resource_levels=resource_levels,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        use_all_maps=use_all_maps,
        maps_to_use=parsed_maps_to_use,
        include_diagnostics=include_diagnostics,
        max_uses_values=parsed_max_uses_values,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # Create eval suite with the dense training missions
    # Build eval simulations for each mission at full resource level (10)
    eval_simulations = []
    for mission in DENSE_TRAINING_MISSIONS:
        env = mission.make_env()
        sim = SimulationConfig(
            suite="dense_curriculum",
            name=f"{mission.name}_eval",
            env=env,
        )
        eval_simulations.append(sim)

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_simulations,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    resource_level: int = 10,
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on dense training missions.

    Args:
        policy_uris: Policy URIs to evaluate
        num_cogs: Number of agents per environment
        resource_level: Resource density level (1-10)
        variants: Optional mission variants

    Returns:
        An EvaluateTool configured for evaluation
    """
    simulations = []

    for mission in DENSE_TRAINING_MISSIONS:
        # Extract map name and other variants
        map_name = None
        other_variants = []
        for variant in mission.variants or []:
            if hasattr(variant, "map_name"):
                map_name = variant.map_name
            else:
                other_variants.append(variant)

        # Create variants with resource reduction
        if resource_level < 10:
            mission_variants = [
                ResourceReductionVariant(level=resource_level, map_name=map_name, seed=42),
                *other_variants,
            ]
        else:
            # Full resources - use original variants
            mission_variants = list(mission.variants) if mission.variants else []

        modified_mission = Mission(
            name=mission.name,
            description=mission.description,
            site=mission.site,
            variants=mission_variants,
            num_cogs=num_cogs,
        )

        env = modified_mission.make_env()
        sim = SimulationConfig(
            suite="dense_curriculum_eval",
            name=f"{mission.name}_rlvl{resource_level}",
            env=env,
        )
        simulations.append(sim)

    return EvaluateTool(
        simulations=simulations,
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission_name: str = "dense_training_4agents",
    resource_level: int = 10,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play a single mission with a policy at a specific resource level.

    Args:
        policy_uri: Optional policy to use
        mission_name: Name of the mission to play (without _level suffix)
        resource_level: Resource density level (1-10, where 10 = full resources)
        num_cogs: Number of agents per environment
        variants: Optional mission variants

    Returns:
        A PlayTool configured for interactive play

    Examples:
        Play with full resources:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.play

        Play with reduced resources (level 5):
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.play resource_level=5

        Play a different map:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.play \\
                mission_name=dense_training_big resource_level=3
    """
    # Find the mission
    mission = None
    for m in DENSE_TRAINING_MISSIONS:
        if m.name == mission_name:
            mission = m
            break

    if mission is None:
        raise ValueError(
            f"Mission '{mission_name}' not found in DENSE_TRAINING_MISSIONS. "
            f"Available missions: {[m.name for m in DENSE_TRAINING_MISSIONS]}"
        )

    # Extract map name and other variants
    map_name = None
    other_variants = []
    for variant in mission.variants or []:
        if hasattr(variant, "map_name"):
            map_name = variant.map_name
        else:
            other_variants.append(variant)

    # Create variants with resource reduction
    if resource_level < 10:
        mission_variants = [
            ResourceReductionVariant(level=resource_level, map_name=map_name, seed=42),
            *other_variants,
        ]
    else:
        # Full resources - use original variants
        mission_variants = list(mission.variants) if mission.variants else []

    modified_mission = Mission(
        name=mission.name,
        description=mission.description,
        site=mission.site,
        variants=mission_variants,
        num_cogs=num_cogs,
    )

    env = modified_mission.make_env()
    sim = SimulationConfig(
        suite="dense_curriculum_play",
        name=f"{mission_name}_rlvl{resource_level}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    run_name: Optional[str] = None,
    num_cogs: int = 4,
    resource_levels: list[int] | None = None,
    use_all_maps: bool = True,
    maps_to_use: Optional[list[str]] = None,
    include_diagnostics: bool = False,
    max_uses_values: Optional[list[int]] = None,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        run_name: Optional run name
        num_cogs: Number of agents per environment (4 or 24)
        resource_levels: Resource levels to include
        use_all_maps: If True, use all maps. If False, only big/small maps
        maps_to_use: Optional list of map names to include (overrides use_all_maps)
        include_diagnostics: If True, add diagnostic missions at full resources
        max_uses_values: Optional list of max_uses values (default: [1, 2, 3, 5, 10, 255])
        heartbeat_timeout: Heartbeat timeout in seconds
        skip_git_check: Whether to skip git check
        additional_args: Additional arguments to pass

    Examples:
        Submit 4-agent training with all maps:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.experiment

        Submit 24-agent training with big/small maps only:
            uv run ./tools/run.py recipes.experiment.cvc.dense_curriculum.experiment \\
                run_name=dense_24agent num_cogs=24 use_all_maps=false
    """
    import json

    if run_name is None:
        map_suffix = "all" if use_all_maps else "big"
        if maps_to_use:
            map_suffix = "_".join([m.replace("dense_training_", "") for m in maps_to_use])
        diag_suffix = "diag" if include_diagnostics else "nodiag"
        run_name = f"dense_{num_cogs}agent_{map_suffix}_{diag_suffix}_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.dense_curriculum.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        f"use_all_maps={str(use_all_maps).lower()}",
        f"include_diagnostics={str(include_diagnostics).lower()}",
        "--gpus=4",
        f"--heartbeat-timeout={heartbeat_timeout}",
    ]

    if resource_levels:
        # Pass as JSON string without spaces to avoid shell parsing issues
        # Use separators to remove spaces so shell won't split the argument
        resource_levels_str = json.dumps(resource_levels, separators=(",", ":"))
        cmd.append(f"resource_levels={resource_levels_str}")

    if maps_to_use:
        # Pass as JSON string without spaces
        maps_to_use_str = json.dumps(maps_to_use, separators=(",", ":"))
        cmd.append(f"maps_to_use={maps_to_use_str}")

    if max_uses_values:
        # Pass as JSON string without spaces
        max_uses_str = json.dumps(max_uses_values, separators=(",", ":"))
        cmd.append(f"max_uses_values={max_uses_str}")

    if skip_git_check:
        cmd.append("--skip-git-check")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching dense curriculum training job: {run_name}")
    print(f"  Agents: {num_cogs}")
    if maps_to_use:
        print(f"  Maps: {', '.join(maps_to_use)}")
    else:
        print(f"  Maps: {'All 4 maps' if use_all_maps else 'Big/Small only'}")
    print(f"  Diagnostics: {'Included' if include_diagnostics else 'Excluded'}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


__all__ = [
    "make_dense_curriculum",
    "train",
    "evaluate",
    "play",
    "experiment",
    "ResourceReductionVariant",
    "MaxUsesVariant",
    "MAP_NAMES",
]
