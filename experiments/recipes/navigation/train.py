#!/usr/bin/env python3
"""
Navigation training recipe for Metta.

This module provides functions for training navigation agents using different
curriculum strategies.
"""

import os
from datetime import datetime
from typing import Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import ValueRange
from metta.map.terrain_from_numpy import TerrainFromNumpy
from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.evals.navigation import make_navigation_eval_suite


def _get_user_identifier() -> str:
    """Get user identifier from USER environment variable."""
    return os.getenv("USER", "unknown")


def _default_run_name() -> str:
    """Generate a robust run name following the pattern: navigation.{user}.{date}.{unique_id}

    Format: navigation.{username}.MMDD-HHMMSS.{git_hash_short} or navigation.{username}.MMDD-HHMMSS
    Example: navigation.alice.0820-143052.a1b2c3d or navigation.alice.0820-143052"""
    user = _get_user_identifier()
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M%S")

    # Try to get git hash (7 chars like CI) for better tracking
    try:
        from metta.common.util.git import get_current_commit

        git_hash = get_current_commit()[:7]
        return f"navigation.{user}.{timestamp}.{git_hash}"
    except Exception:
        # Fallback: use timestamp
        return f"navigation.{user}.{timestamp}"


def make_env(num_agents: int = 4) -> MettaGridConfig:
    """Create a navigation environment configuration."""
    nav = eb.make_navigation(num_agents=num_agents)

    nav.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance_map=TerrainFromNumpy.Config(
            agents=1,
            objects={"altar": 10},
            dir="varied_terrain/dense_large",
        ),
    )
    return nav


def make_curriculum(
    nav_env: Optional[MettaGridConfig] = None, use_learning_progress: bool = True
) -> CurriculumConfig:
    """Create a navigation curriculum configuration.

    Args:
        nav_env: Navigation environment configuration
        use_learning_progress: Whether to use learning progress algorithm (default: True)
    """
    nav_env = nav_env or make_env()

    # make a set of training tasks for navigation
    nav_tasks = cc.bucketed(nav_env)

    # dense reward tasks
    dense_tasks = cc.bucketed(nav_env)
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.5, 1.0])
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart_max", [1, 2])

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    dense_tasks.add_bucket(
        "game.map_builder.instance_map.objects.altar", [ValueRange.vr(3, 50)]
    )

    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"altar": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [ValueRange.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [ValueRange.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [ValueRange.vr(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    if use_learning_progress:
        # Use the updated to_curriculum method that defaults to learning progress
        return nav_tasks.to_curriculum()
    else:
        # Create curriculum without learning progress algorithm (random sampling)
        return CurriculumConfig(task_generator=nav_tasks)


def train(
    run: Optional[str] = None,
    curriculum: Optional[CurriculumConfig] = None,
    use_learning_progress: bool = True,
) -> TrainTool:
    """Create a navigation training tool.

    Args:
        run: Run name (auto-generated if not provided)
        curriculum: Curriculum configuration (auto-generated if not provided)
        use_learning_progress: Whether to use learning progress algorithm
    """
    # Generate structured run name if not provided
    if run is None:
        run = _default_run_name()

    # Create curriculum if not provided
    if curriculum is None:
        curriculum = make_curriculum(use_learning_progress=use_learning_progress)

    trainer_cfg = TrainerConfig(
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=make_navigation_eval_suite(),
        ),
    )

    # Create the training tool
    train_tool = TrainTool(trainer=trainer_cfg)

    return train_tool


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Create a navigation play tool."""
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Create a navigation replay tool."""
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def eval() -> SimTool:
    """Create a navigation evaluation tool."""
    return SimTool(simulations=make_navigation_eval_suite())
