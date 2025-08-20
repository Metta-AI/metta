import os
import uuid
from datetime import datetime
from typing import Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.map.mapgen import MapGen
from metta.map.terrain_from_numpy import TerrainFromNumpy
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.evals.navigation import make_navigation_eval_suite


def _get_user_identifier() -> str:
    """Get user identifier, trying multiple environment variables."""
    # Try common user environment variables across platforms
    for var in ["USER", "USERNAME", "LOGNAME"]:
        user = os.getenv(var)
        if user and user.strip():
            return user.strip()
    return "unknown"


def _default_run_name() -> str:
    """Generate a robust run name following the pattern: navigation.{user}.{date}.{unique_id}

    Format: navigation.{username}.MMDD-HHMM.{uuid}
    Example: navigation.alice.0820-1430.f4b2c8
    """
    user = _get_user_identifier()
    now = datetime.now()
    # Use 6-char UUID
    unique_id = str(uuid.uuid4())[:6]
    return f"navigation.{user}.{now.strftime('%m%d-%H%M')}.{unique_id}"


def make_env(num_agents: int = 4) -> EnvConfig:
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


def make_curriculum(nav_env: Optional[EnvConfig] = None) -> CurriculumConfig:
    nav_env = nav_env or make_env()

    # make a set of training tasks for navigation
    nav_tasks = cc.tasks(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    nav_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    nav_tasks.add_bucket("game.map_builder.instance_map.objects.altar", [vr.vr(3, 50)])

    # Additional curriculum buckets from sparse and sparse_bucketed configurations
    nav_tasks.add_bucket("game.max_steps", [vr.vr(100, 2000)])
    nav_tasks.add_bucket("game.map_builder.width", [vr.vr(60, 300)])
    nav_tasks.add_bucket("game.map_builder.height", [vr.vr(60, 300)])

    return cc.curriculum(nav_tasks, num_tasks=1000)


def train(
    run: Optional[str] = None, curriculum: Optional[CurriculumConfig] = None
) -> TrainTool:
    # Generate structured run name if not provided
    if run is None:
        run = _default_run_name()

    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_navigation_eval_suite(),
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        run=run,
    )


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def eval() -> SimTool:
    return SimTool(simulations=make_navigation_eval_suite())
