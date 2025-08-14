import metta.mettagrid.config.envs as eb
from experiments import arena, navigation
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.tools.play import PlayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

########################################################
# Arena
########################################################

# Obstacles
obstacles_env = eb.make_navigation(num_agents=1)
obstacles_env.game.map_builder = AsciiMapBuilder.Config.from_uri(
    "configs/env/mettagrid/maps/navigation/obstacles0.map",
)
obstacles_env.game.max_steps = 100


def train() -> TrainTool:
    env = navigation.make_env()
    env.game.max_steps = 100
    cfg = navigation.train(
        run="local.daveey.1",
        curriculum=navigation.make_curriculum(env),
    )
    return cfg


def play() -> PlayTool:
    env = arena.make_evals()[0].env
    env.game.max_steps = 100
    # env.game.agent.initial_inventory["battery_red"] = 10
    cfg = arena.play(env)
    return cfg


def evaluate() -> SimTool:
    cfg = arena.evaluate(policy_uri="wandb://run/local.daveey.1")

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg
