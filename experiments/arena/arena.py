import metta.cogworks.curriculum as cc
import metta.mettagrid.config.builder as eb
import softmax.softmax as softmax
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SingleEnvSimulationConfig
from tools.play import PlayToolConfig
from tools.replay import ReplayToolConfig
from tools.utils import calculate_default_num_workers

arena = eb.arena(num_agents=24)

# disable swap
arena.game.actions.swap.enabled = False


def make_trainer_cfg(run: str, data_dir: str) -> TrainerConfig:
    # make a set of training tasks for the arena
    arena_tasks = cc.tasks(arena)

    arena_tasks.add_bucket("game.map_builder.num_agents", [1, 2, 3, 4, 6, 24])
    arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40, 50])
    arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40, 50])

    for item in arena.game.inventory_item_names:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, vr.vr(0, 1.0)]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    curriculum_cfg = cc.curriculum(arena_tasks, num_tasks=4)

    print(curriculum_cfg.model_dump_json(indent=2))

    trainer_cfg = TrainerConfig()
    trainer_cfg.simulation.replay_dir = f"s3://softmax-public/replays/{run}"
    trainer_cfg.num_workers = calculate_default_num_workers(is_serial=True)
    trainer_cfg.checkpoint.checkpoint_dir = f"{data_dir}/{run}/checkpoints"
    trainer_cfg.curriculum = curriculum_cfg

    return trainer_cfg


def make_replay_tool_cfg() -> ReplayToolConfig:
    eval_env = arena.copy()
    eval_env.game.max_steps = 100
    return ReplayToolConfig(
        sim=SingleEnvSimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        system=SystemConfig.MacBookPro(),
        wandb=softmax.wandb_config(run="arena_replay"),
    )


def make_play_tool_cfg() -> PlayToolConfig:
    eval_env = arena.copy()
    eval_env.game.max_steps = 100
    return PlayToolConfig(
        sim=SingleEnvSimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        system=SystemConfig.MacBookPro(),
        wandb=softmax.wandb_config(run="arena_replay"),
    )
