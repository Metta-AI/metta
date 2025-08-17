<<<<<<< HEAD
<<<<<<< HEAD
=======
import metta.cogworks.curriculum as cc
=======
>>>>>>> 823b1bb0e (cp)
import metta.mettagrid.config.envs as eb
from experiments import arena
from metta.mettagrid.map_builder import AsciiMapBuilderConfig

########################################################
# Arena
########################################################

# Obstacles
obstacles_env = eb.make_navigation(num_agents=1)
obstacles_env.game.map_builder = AsciiMapBuilderConfig.from_uri(
    "configs/env/mettagrid/maps/navigation/obstacles0.map",
)
obstacles_env.game.max_steps = 100

# def tool_cfg_replay() -> ReplayTool:
#     eval_env = obstacles.model_copy()
#     eval_env.game.max_steps = 100
#     return ReplayTool(
#         sim=SimulationConfig(
#             env=eval_env,
#             name="arena",
#         ),
#         open_browser_on_start=True,
#         wandb=softmax.wandb_config(run="arena_replay"),
#     )


def train():
    env = arena.make_env()
    env.game.max_steps = 100
    cfg = arena.train(
        run="local.daveey.1",
        curriculum=arena.make_curriculum(env),
    )
    return cfg


<<<<<<< HEAD
def tool_cfg_train(run: str = "daveey-arena") -> TrainTool:
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
>>>>>>> 0dd0ae4fb (cp)
=======
play = arena.play
>>>>>>> 823b1bb0e (cp)
