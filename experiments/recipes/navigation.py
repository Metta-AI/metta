import os
from datetime import datetime
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import metta.mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import TerrainFromNumpy
from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.loss.loss_config import LossConfig
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


def make_mettagrid(num_agents: int = 1, num_instances: int = 4) -> MettaGridConfig:
    nav = eb.make_navigation(num_agents=num_agents * num_instances)

    nav.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
        instance_map=TerrainFromNumpy.Config(
            agents=num_agents,
            objects={"altar": 10},
            dir="varied_terrain/dense_large",
        ),
    )
    return nav


def make_curriculum(nav_env: Optional[MettaGridConfig] = None) -> CurriculumConfig:
    nav_env = nav_env or make_mettagrid()

    # make a set of training tasks for navigation
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance_map.objects.altar", [Span(3, 50)])

    # sparse environments are just random maps
    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"altar": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [Span(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    return CurriculumConfig(task_generator=nav_tasks)


def train(
    run: Optional[str] = None, curriculum: Optional[CurriculumConfig] = None
) -> TrainTool:
    # Generate structured run name if not provided
    if run is None:
        run = _default_run_name()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_navigation_eval_suite(),
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        run=run,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    simulations = simulations or make_navigation_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
