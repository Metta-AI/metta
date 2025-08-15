<<<<<<< HEAD
=======
import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
import softmax.softmax as softmax
import yaml
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.mettagrid.map_builder import AsciiMapBuilderConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from tools.play import PlayTool
from tools.replay import ReplayToolConfig

########################################################
# Environments
########################################################

# Arena
arena = eb.make_arena(num_agents=24)
arena.game.actions.swap.enabled = False

# Obstacles
obstacles = eb.make_nav(num_agents=1)
obstacles.game.map_builder = AsciiMapBuilderConfig.from_uri(
    "configs/env/mettagrid/maps/navigation/obstacles0.map",
)
obstacles.game.max_steps = 100

# Varied Terrain
varied_terrain = eb.make_nav(num_agents=4)
varied_terrain.game.objects["altar"].cooldown = 1000

# varied_terrain.game.map_builder = TerrainFromNumpyConfig(
#         agents=1,
#         objects={"altar": 10},
#         dir="varied_terrain/dense_large",
#     ).pack()

# varied_terrain.game.map_builder = MapGenConfig(
#     instances=4,
#     border_width=6,
#     instance_border_width=3,
#     root="hallway",
#     # instance_map=TerrainFromNumpyConfig(
#     #     agents=1,
#     #     objects={"altar": 10},
#     #     dir="varied_terrain/dense_large",
#     # )
# )

########################################################
# Tools
########################################################


def tool_cfg_replay() -> ReplayToolConfig:
    eval_env = obstacles.model_copy()
    eval_env.game.max_steps = 100
    return ReplayToolConfig(
        sim=SimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        wandb=softmax.wandb_config(run="arena_replay"),
    )


def tool_cfg_play() -> PlayTool:
    eval_env = varied_terrain.model_copy()
    eval_env.game.max_steps = 100
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="arena",
        ),
        open_browser_on_start=True,
        wandb=softmax.wandb_config(run="arena_replay"),
    )


def tool_cfg_train() -> TrainTool:
    run = "daveey-test-run-3"

    # make a set of training tasks for the arena
    arena_tasks = cc.tasks(arena)

    # arena_tasks.add_bucket("game.map_builder.agents", [1, 2, 3, 4, 6, 24])
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
        run="daveey-arena",
        policy_architecture=yaml.safe_load(open("configs/agent/fast.yaml")),
    )
>>>>>>> 0dd0ae4fb (cp)
