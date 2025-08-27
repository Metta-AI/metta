import os
from datetime import datetime
from typing import Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.map.mapgen import MapGen
from metta.map.terrain_from_numpy import TerrainFromNumpy
from metta.mettagrid.mettagrid_config import EnvConfig
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
    Example: navigation.alice.0820-143052.a1b2c3d or navigation.alice.0820-143052
    """
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
    nav_tasks = cc.bucketed(nav_env)

    # dense reward tasks
    dense_tasks = cc.bucketed(nav_env)
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.5, 1.0])
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart_max", [1, 2])

    # sparse reward tasks
    sparse_tasks = cc.bucketed(nav_env)
    sparse_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.0])
    sparse_tasks.add_bucket("game.agent.rewards.inventory.heart_max", [1])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    return nav_tasks.to_curriculum()


def train(
    run: Optional[str] = None,
    curriculum: Optional[CurriculumConfig] = None,
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

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def eval() -> SimTool:
    return SimTool(simulations=make_navigation_eval_suite())
