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


play = arena.play
