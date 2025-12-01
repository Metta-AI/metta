"""Random map curriculum for CoGs vs Clips.

Uses procedural random map generation with mettagrid.mapgen.scenes.random.Random.Config,
with curriculum bucketing over map dimensions and object counts.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.sites import HELLO_WORLD
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config import vibes as vibes_module
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.random import Random
from recipes.experiment import cogs_v_clips


def make_random_maps_curriculum(
    num_cogs: int = 20,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
    heart_buckets: bool = False,
    resource_buckets: bool = False,
) -> CurriculumConfig:
    """Create a curriculum with randomly generated maps.

    Args:
        num_cogs: Number of agents per environment
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        variants: Optional mission variants to apply

    Returns:
        A CurriculumConfig with learning progress algorithm across map sizes and object counts
    """
    # Create base mission with random map generation
    mission = Mission(
        name="random_maps_training",
        description="Random procedural maps with varying dimensions and object counts",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
    )

    # Create base environment
    base_env = mission.make_env()

    # Use only first 13 vibes from TRAINING_VIBES
    # (excludes: assembler, chest, wall, red-heart)
    from mettagrid.config import vibes as vibes_module

    first_13_vibes = [v.name for v in vibes_module.VIBES[:13]]
    base_env.game.vibe_names = first_13_vibes
    if base_env.game.actions.change_vibe:
        base_env.game.actions.change_vibe.number_of_vibes = 13

    # Replace map builder with random map generator
    # Random.Config has too_many_is_ok=True, so it will cap objects to available space
    base_env.game.map_builder = MapGen.Config(
        width=80,  # Default, will be bucketed
        height=80,  # Default, will be bucketed
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 10,  # Default, will be bucketed
                "charger": 5,  # Default, will be bucketed
                "chest": 2,  # Default, will be bucketed
                "carbon_extractor": 5,  # Default, will be bucketed
                "oxygen_extractor": 5,  # Default, will be bucketed
                "germanium_extractor": 5,  # Default, will be bucketed
                "silicon_extractor": 5,  # Default, will be bucketed
            },
            too_many_is_ok=True,  # Automatically caps to available space
        ),
    )

    # Create bucketed tasks
    tasks = cc.bucketed(base_env)

    # Bucket over map dimensions (20x20 to 150x150)
    tasks.add_bucket("game.map_builder.width", [Span(30, 100)])
    tasks.add_bucket("game.map_builder.height", [Span(30, 100)])

    # Bucket over object counts (sparse to dense)
    # Using wide ranges that scale from small maps to large maps
    # too_many_is_ok=True ensures we don't error on small maps
    tasks.add_bucket("game.map_builder.instance.objects.assembler", [Span(10, 30)])
    tasks.add_bucket("game.map_builder.instance.objects.charger", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.chest", [Span(10, 30)])
    tasks.add_bucket("game.map_builder.instance.objects.carbon_extractor", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.oxygen_extractor", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.germanium_extractor", [Span(10, 40)])
    tasks.add_bucket("game.map_builder.instance.objects.silicon_extractor", [Span(10, 50)])

    # Bucket over extractor max_uses (resource scarcity)
    # 0 = unlimited, higher = limited resource
    tasks.add_bucket("game.objects.carbon_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.oxygen_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.germanium_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.silicon_extractor.max_uses", [1, 3, 8, 10, 20])

    # Standard curriculum buckets
    tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500, 2000, 3000, 4000])

    if heart_buckets:
        tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])
    if resource_buckets:
        tasks.add_bucket("game.agent.rewards.stats.carbon.gained", [0.0, 0.01])
        tasks.add_bucket("game.agent.rewards.stats.oxygen.gained", [0.0, 0.01])
        tasks.add_bucket("game.agent.rewards.stats.germanium.gained", [0.0, 0.01])
        tasks.add_bucket("game.agent.rewards.stats.silicon.gained", [0.0, 0.01])

        stats_max_cap = 0.5
        tasks.add_bucket("game.agent.rewards.stats_max.carbon.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.oxygen.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.germanium.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.silicon.gained", [stats_max_cap])

    # Configure learning progress algorithm
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=3000,
            max_slice_axes=4,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return tasks.to_curriculum(
        num_active_tasks=2000,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 20,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    heart_buckets = False,
    resource_buckets = False,
) -> TrainTool:
    """Create a training tool for random maps curriculum.

    Args:
        num_cogs: Number of agents per environment
        curriculum: Optional curriculum configuration
        enable_detailed_slice_logging: Enable detailed logging
        variants: Optional mission variants for training
        eval_variants: Optional mission variants for evaluation
        eval_difficulty: Difficulty level for evaluation

    Returns:
        A TrainTool configured with the random maps curriculum

    Examples:
        Train with 4 agents:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.train num_cogs=4

        Train with 8 agents:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.train num_cogs=8
    """
    resolved_curriculum = curriculum or make_random_maps_curriculum(
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        heart_buckets=heart_buckets,
        resource_buckets=resource_buckets,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # Use standard CVC eval suite
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
    num_cogs: int = 20,
    difficulty: str | None = "standard",
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on standard CVC missions.

    Args:
        policy_uris: Policy URIs to evaluate
        num_cogs: Number of agents per environment
        difficulty: Difficulty variant
        variants: Optional mission variants

    Returns:
        An EvaluateTool configured for evaluation
    """
    return EvaluateTool(
        simulations=cogs_v_clips.make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def play_sparse(
    policy_uri: Optional[str] = None,
    num_cogs: int = 20,
    room_size: int = 80,
) -> PlayTool:
    """Play on a sparse randomly generated map (minimum objects).

    Args:
        policy_uri: Optional policy to use
        num_cogs: Number of agents
        room_size: Map width and height (creates square map)

    Returns:
        A PlayTool configured for sparse map play

    Examples:
        Play small sparse map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_sparse room_size=30

        Play large sparse map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_sparse room_size=150
    """
    mission = Mission(
        name="random_maps_sparse",
        description=f"Sparse random map {room_size}x{room_size}",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
    )

    env = mission.make_env()

    # Use only first 13 vibes from TRAINING_VIBES
    from mettagrid.config import vibes as vibes_module

    env.game.vibe_names = [v.name for v in vibes_module.VIBES[:13]]
    if env.game.actions.change_vibe:
        env.game.actions.change_vibe.number_of_vibes = 13

    env.game.map_builder = MapGen.Config(
        width=room_size,
        height=room_size,
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 1,
                "charger": 1,
                "chest": 1,
                "carbon_extractor": 2,
                "oxygen_extractor": 2,
                "germanium_extractor": 2,
                "silicon_extractor": 2,
            },
            too_many_is_ok=True,
        ),
    )

    sim = SimulationConfig(
        suite="random_maps",
        name=f"sparse_{room_size}x{room_size}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_dense(
    policy_uri: Optional[str] = None,
    num_cogs: int = 20,
    room_size: int = 100,
) -> PlayTool:
    """Play on a dense randomly generated map (maximum objects).

    Args:
        policy_uri: Optional policy to use
        num_cogs: Number of agents
        room_size: Map width and height (creates square map)

    Returns:
        A PlayTool configured for dense map play

    Examples:
        Play small dense map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_dense room_size=30

        Play large dense map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_dense room_size=150
    """
    mission = Mission(
        name="random_maps_dense",
        description=f"Dense random map {room_size}x{room_size}",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
    )

    env = mission.make_env()

    # Restrict vibes to only heart_b and default
    env.game.vibe_names = [v.name for v in vibes_module.VIBES[:13]]
    if env.game.actions.change_vibe:
        env.game.actions.change_vibe.number_of_vibes = len(env.game.vibe_names)

    env.game.map_builder = MapGen.Config(
        width=room_size,
        height=room_size,
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 50,
                "charger": 50,
                "chest": 50,
                "carbon_extractor": 50,
                "oxygen_extractor": 50,
                "germanium_extractor": 50,
                "silicon_extractor": 50,
            },
            too_many_is_ok=True,  # Will cap to available space on small maps
        ),
    )

    sim = SimulationConfig(
        suite="random_maps",
        name=f"dense_{room_size}x{room_size}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    run_name: Optional[str] = None,
    num_cogs: int = 20,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    additional_args: Optional[list[str]] = None,
    heart_buckets: bool = False,
    resource_buckets: bool = False,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        run_name: Optional run name
        num_cogs: Number of agents per environment
        heartbeat_timeout: Heartbeat timeout in seconds
        skip_git_check: Whether to skip git check
        additional_args: Additional arguments to pass

    Examples:
        Submit training:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.experiment

        Submit with custom name:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.experiment \\
                run_name=random_maps_4agent
    """
    if run_name is None:
        run_name = f"cvc_random_maps_{num_cogs}agent_heartbuckets_{heart_buckets}_resourcebuckets_{resource_buckets}_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.cvc_random_maps.train",
        f"run={run_name}-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        "--heartbeat-timeout-seconds=3600",
    ]

    if skip_git_check:
        cmd.append("--skip-git-check")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching random maps training job: {run_name}")
    print(f"  Agents: {num_cogs}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


__all__ = [
    "make_random_maps_curriculum",
    "train",
    "evaluate",
    "play_sparse",
    "play_dense",
    "experiment",
]

if __name__ == "__main__":
    experiment(heart_buckets=False, resource_buckets=False)
    experiment(heart_buckets=True, resource_buckets=False)
    experiment(heart_buckets=True, resource_buckets=True)
    experiment(heart_buckets=False, resource_buckets=True)
