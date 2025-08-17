import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
import softmax.softmax as softmax
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.map.mapgen import MapGenConfig
from metta.map.terrain_from_numpy import TerrainFromNumpyConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool

from experiments.evals.navigation import make_navigation_eval_suite

nav = eb.make_navigation(num_agents=4)

nav.game.map_builder = MapGenConfig(
    instances=4,
    border_width=6,
    instance_border_width=3,
    instance_map=TerrainFromNumpyConfig(
        agents=1,
        objects={"altar": 10},
        dir="varied_terrain/dense_large",
    ),
)
nav_tasks = cc.tasks(nav)

maps = ["terrain_maps_nohearts"]
for size in ["large", "medium", "small"]:
    for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
        maps.append(f"varied_terrain/{terrain}_{size}")

nav_tasks.add_bucket("game.map_builder.instance_map.params.dir", maps)
nav_tasks.add_bucket(
    "game.map_builder.instance_map.params.objects.altar", [vr.vr(3, 50)]
)

# TODO #dehydration
# add /env/mettagrid/curriculum/navigation/subcurricula/sparse
# add /env/mettagrid/navigation/training/sparse_bucketed: 1

curriculum_cfg = cc.curriculum(nav_tasks, num_tasks=4)


def train(run: str) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum_cfg,
        evaluation=EvaluationConfig(
            replay_dir=f"s3://softmax-public/replays/{run}",
            evaluate_remote=False,
            evaluate_local=True,
            simulations=make_navigation_eval_suite(),
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        wandb=softmax.wandb_config(run=run),
        run=run,
    )


def play() -> PlayTool:
    return PlayTool(
        sim=SimulationConfig(
            env=nav,
            name="navigation",
        ),
        wandb=softmax.wandb_config(run="navigation.play"),
    )
