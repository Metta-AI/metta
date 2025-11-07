# ruff: noqa: E501
try:
    import metta.tools.eval
    import metta.tools.play
    import metta.tools.replay
    import metta.tools.train

    import experiments.recipes
except Exception as e:
    print(f"Error importing: {e}")
    print("Run `metta install sandbox --force` to fix this.")
    raise

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.{{ USER }}.train`
# The VSCode "Run and Debug" section supports options to run these functions.


def train() -> metta.tools.train.TrainTool:
    env = experiments.recipes.arena.mettagrid(num_agents=24)
    env.game.max_steps = 100
    cfg = experiments.recipes.arena.train(
        curriculum=experiments.recipes.arena.make_curriculum(env),
    )
    assert cfg.evaluator is not None
    # When we're using this file, we training locally on code that's likely not to be checked in, let alone pushed.
    # So remote evaluation probably doesn't make sense.
    cfg.evaluator.evaluate_remote = False
    cfg.evaluator.evaluate_local = True
    return cfg


def play() -> metta.tools.play.PlayTool:
    env = experiments.recipes.arena.mettagrid(num_agents=24)
    env.game.max_steps = 100
    cfg = experiments.recipes.arena.play(policy_uri=None)
    return cfg


def replay() -> metta.tools.replay.ReplayTool:
    env = experiments.recipes.arena.mettagrid(num_agents=24)
    env.game.max_steps = 100
    cfg = experiments.recipes.arena.replay(policy_uri=None)
    # cfg.policy_uri = "s3://your-bucket/checkpoints/daveey.combat.lpsm.8x4/daveey.combat.lpsm.8x4:v42.pt"
    return cfg


def evaluate(
    policy_uri: str = "s3://your-bucket/checkpoints/local.{{ USER }}.1/local.{{ USER }}.1:v10.pt",
) -> metta.tools.eval.EvaluateTool:
    cfg = experiments.recipes.arena.evaluate(policy_uris=policy_uri)

    return cfg
