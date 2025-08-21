from experiments.recipes import arena
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool


def train() -> TrainTool:
    cfg = arena.train(curriculum=arena.make_curriculum(arena.make_env()))

    # CI-friendly runtime: tiny steps and small batches.
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"

    # Mirror CI overrides from .github/workflows/test-mettascope.yml
    cfg.trainer.total_timesteps = 16
    cfg.trainer.minibatch_size = 8
    cfg.trainer.batch_size = 1536
    cfg.trainer.bptt_horizon = 8
    cfg.trainer.forward_pass_minibatch_target_size = 192
    cfg.trainer.checkpoint.checkpoint_interval = 0
    cfg.trainer.checkpoint.wandb_checkpoint_interval = 0
    if cfg.trainer.evaluation is not None:
        cfg.trainer.evaluation.evaluate_interval = 0

    # Name the run for CI readability.
    cfg.run = "smoke_test"
    return cfg


def replay() -> ReplayTool:
    env = arena.make_env()
    env.game.max_steps = 100
    cfg = arena.replay(env)
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    # Ensure label is present as expected by frontend/tests.
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def play() -> PlayTool:
    env = arena.make_env()
    env.game.max_steps = 100
    cfg = arena.play(env)
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def play_null() -> PlayTool:
    cfg = play()
    # Use mock policy (Simulation will create mock when policy_uri is None).
    cfg.policy_uri = None
    return cfg
