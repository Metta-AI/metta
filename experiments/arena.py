from typing import Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
import softmax.softmax as softmax
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool


def make_env(num_agents: int = 24) -> EnvConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    arena_env.game.actions.swap.enabled = False
    return arena_env


def make_curriculum(arena_env: Optional[EnvConfig] = None) -> CurriculumConfig:
    arena_env = arena_env or make_env()

    # make a set of training tasks for the arena
    arena_tasks = cc.tasks(arena_env)

    arena_tasks.add_bucket("game.map_builder.root.params.agents", [1, 2, 3, 4, 6, 24])
    arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40, 50])
    arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40, 50])

    for item in arena_env.game.inventory_item_names:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, vr.vr(0, 1.0)]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])
    return cc.curriculum(arena_tasks, num_tasks=4)


def train(run: str, curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            replay_dir=f"s3://softmax-public/replays/{run}",
            evaluate_remote=False,
            evaluate_local=True,
            simulations=[
                SimulationConfig(
                    name="arena/basic", env=eb.make_arena(num_agents=24, combat=False)
                ),
                SimulationConfig(
                    name="arena/combat", env=eb.make_arena(num_agents=24, combat=True)
                ),
            ],
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        wandb=softmax.wandb_config(run=run),
        run=run,
    )


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="arena",
        ),
        wandb=softmax.wandb_config(run="arena.play"),
    )
