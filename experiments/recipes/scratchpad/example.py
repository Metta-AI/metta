from experiments.recipes import arena
from softmax.training.tools.play import PlayTool
from softmax.training.tools.replay import ReplayTool
from softmax.training.tools.sim import SimTool
from softmax.training.tools.train import TrainTool

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.{{ USER }}.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> TrainTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.train(
        curriculum=arena.make_curriculum(env),
    )
    assert cfg.evaluator is not None
    # When we're using this file, we training locally on code that's likely not to be checked in, let alone pushed.
    # So remote evaluation probably doesn't make sense.
    cfg.evaluator.evaluate_remote = False
    cfg.evaluator.evaluate_local = True
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
    # cfg.policy_uri = "s3://your-bucket/checkpoints/daveey.combat.lpsm.8x4/daveey.combat.lpsm.8x4:v42.pt"
    return cfg


def evaluate(
    policy_uri: str = "s3://your-bucket/checkpoints/local.{{ USER }}.1/local.{{ USER }}.1:v10.pt",
) -> SimTool:
    cfg = arena.evaluate(policy_uri=policy_uri)

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="s3://your-bucket/checkpoints/daveey.combat.lpsm.8x4/daveey.combat.lpsm.8x4:v42.pt")
    return cfg
