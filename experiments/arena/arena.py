import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
import softmax.softmax as softmax
import yaml
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from tools.replay import ReplayToolConfig

arena = eb.make_arena(num_agents=24)
arena.game.actions.swap.enabled = False

# make a set of training tasks for the arena
arena_tasks = cc.tasks(arena)

# arena_tasks.add_bucket("game.map_builder.agents", [1, 2, 3, 4, 6, 24])
arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40, 50])
arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40, 50])

for item in arena.game.inventory_item_names:
    arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, vr.vr(0, 1.0)])
    arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

# enable or disable attacks. we use cost instead of 'enabled'
# to maintain action space consistency.
arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

curriculum_cfg = cc.curriculum(arena_tasks, num_tasks=4)


def make_replay_tool_cfg() -> ReplayToolConfig:
    eval_env = arena.model_copy()
    eval_env.game.max_steps = 100
    return ReplayToolConfig(
        sim=SimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        system=SystemConfig(),
        wandb=softmax.wandb_config(run="arena_replay"),
    )


def make_play_tool() -> PlayTool:
    eval_env = arena.model_copy()
    eval_env.game.max_steps = 100
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        system=SystemConfig(),
        wandb=softmax.wandb_config(run="arena_replay"),
    )


def make_train_tool(run: str) -> TrainTool:
    trainer_cfg = TrainerConfig()
    trainer_cfg.curriculum = curriculum_cfg
    trainer_cfg.evaluation = EvaluationConfig(
        replay_dir=f"s3://softmax-public/replays/{run}",
        evaluate_remote=False,
        evaluate_local=True,
        simulations=[
            SimulationConfig(name="arena", env=arena),
        ],
    )

    return TrainTool(
        trainer=trainer_cfg,
        wandb=softmax.wandb_config(run=run),
        run=run,
        policy_architecture=yaml.safe_load(open("configs/agent/fast.yaml")),
    )
