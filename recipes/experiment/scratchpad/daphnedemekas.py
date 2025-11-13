from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from experiments.recipes.cogs_v_clips import generalized_terrain, foraging, facilities

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.daphnedemekas.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> TrainTool:
    return generalized_terrain.train(curriculum_style="multi_agent_pairs_bases")


def play() -> PlayTool:
    return generalized_terrain.play(
        curriculum_style="multi_agent_singles_uniform",
    )


# def play() -> PlayTool:
#     env = arena.make_evals()[0].env
#     env.game.max_steps = 100
#     cfg = arena.play(env)
#     return cfg


# def replay() -> ReplayTool:
#     env = arena.make_mettagrid()
#     env.game.max_steps = 100
#     cfg = arena.replay(env)
#     # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
#     return cfg


# def evaluate(run: str = "local.daphnedemekas.1") -> SimTool:
#     cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

#     # If your run doesn't exist, try this:
#     # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
#     return cfg
