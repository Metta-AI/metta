from experiments.recipes import arena
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.{{ USER }}.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> TrainTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.train(
        curriculum=arena.make_curriculum(env),
    )
    assert cfg.trainer.evaluation is not None
    # When we're using this file, we training locally on code that's likely not to be checked in, let alone pushed.
    # So remote evaluation probably doesn't make sense.
    cfg.trainer.evaluation.evaluate_remote = False
    cfg.trainer.evaluation.evaluate_local = True
    return cfg


def play() -> PlayTool:
    env = arena.make_evals()[0].env
    env.game.max_steps = 100
    cfg = arena.play(env)
    return cfg


def replay() -> ReplayTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.replay(env)
    # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
    return cfg


def evaluate(run: str = "local.{{ USER }}.1") -> SimTool:
    cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg
