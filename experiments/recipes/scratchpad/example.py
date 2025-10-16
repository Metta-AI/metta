from experiments.recipes import arena
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.{{ USER }}.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> TrainTool:
    env = arena.mettagrid(num_agents=24)
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
    env = arena.mettagrid(num_agents=24)
    env.game.max_steps = 100
    cfg = arena.play(policy_uri=None)
    return cfg


def replay() -> ReplayTool:
    env = arena.mettagrid(num_agents=24)
    env.game.max_steps = 100
    cfg = arena.replay(policy_uri=None)
    return cfg


def evaluate(
    policy_uri: str = "s3://your-bucket/checkpoints/local.{{ USER }}.1/local.{{ USER }}.1:v10.pt",
) -> EvaluateTool:
    cfg = arena.evaluate(policy_uris=policy_uri)
    return cfg
