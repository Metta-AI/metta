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
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.evals.navigation import make_navigation_eval_suite


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

    # TODO #dehydration
    # add /env/mettagrid/curriculum/navigation/subcurricula/sparse
    # add /env/mettagrid/navigation/training/sparse_bucketed: 1

    return cc.curriculum(nav_tasks, num_tasks=1000)


def train(run: str, curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_navigation_eval_suite(),
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        # TODO #dehydration - get rid of run in here
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
