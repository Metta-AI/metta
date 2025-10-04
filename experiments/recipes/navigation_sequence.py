import os
from datetime import datetime
from typing import Optional

import gitta
import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import TerrainFromNumpy
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from experiments.evals.navigation_sequence import make_navigation_sequence_eval_suite


def _get_user_identifier() -> str:
    """Get user identifier from USER environment variable."""
    return os.getenv("USER", "unknown")


def _default_run_name() -> str:
    """Generate a robust run name following the pattern: navigation_sequence.{user}.{date}.{unique_id}

    Format: navigation_sequence.{username}.MMDD-HHMMSS.{git_hash_short} or navigation_sequence.{username}.MMDD-HHMMSS
    Example: navigation_sequence.alice.0820-143052.a1b2c3d or navigation_sequence.alice.0820-143052"""
    user = _get_user_identifier()
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M%S")

    # Try to get git hash (7 chars like CI) for better tracking
    try:
        git_hash = gitta.get_current_commit()[:7]
        return f"navigation_sequence.{user}.{timestamp}.{git_hash}"
    except Exception:
        # Fallback: use timestamp
        return f"navigation_sequence.{user}.{timestamp}"


def make_env(num_agents: int = 4) -> MettaGridConfig:
    nav = eb.make_navigation_sequence(num_agents=num_agents)

    nav.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance=TerrainFromNumpy.Config(
            agents=1,
            objects={"altar": 15, "mine_red": 15, "generator_red": 15},
            dir="varied_terrain/dense_large",
            remove_altars=True,
        ),
    )
    return nav


def make_curriculum(
    nav_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    nav_env = nav_env or make_env()

    # make a set of training tasks for navigation
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance.objects.altar", [Span(15, 50)])
    dense_tasks.add_bucket("game.map_builder.instance.objects.mine_red", [Span(15, 50)])
    dense_tasks.add_bucket(
        "game.map_builder.instance.objects.generator_red", [Span(15, 50)]
    )
    dense_tasks.add_bucket("game.objects.altar.initial_resource_count", [0, 1])
    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"altar": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [Span(5, 25)])
    sparse_tasks.add_bucket("game.map_builder.objects.mine_red", [Span(5, 25)])
    sparse_tasks.add_bucket("game.map_builder.objects.generator_red", [Span(5, 25)])
    sparse_tasks.add_bucket("game.objects.altar.initial_resource_count", [0, 1])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=3,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return nav_tasks.to_curriculum(algorithm_config=algorithm_config)


def train(
    run: Optional[str] = None,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    # Generate structured run name if not provided
    if run is None:
        run = _default_run_name()

    resolved_curriculum = curriculum or make_curriculum(
        algorithm_config=LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=3,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(simulations=make_navigation_sequence_eval_suite())

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
        run=run,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            suite="navigation_sequence",
            env=eval_env,
            name="eval",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            suite="navigation_sequence",
            env=eval_env,
            name="eval",
        ),
    )


def eval() -> SimTool:
    return SimTool(simulations=make_navigation_sequence_eval_suite())
