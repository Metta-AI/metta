from experiments.recipes import arena
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig

# CI-friendly runtime: tiny steps and small batches.


def train() -> TrainTool:
    cfg = arena.train(curriculum=arena.make_curriculum(arena.mettagrid()))

    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.trainer.total_timesteps = 16  # Minimal steps for smoke testing.
    cfg.trainer.minibatch_size = 8
    cfg.trainer.batch_size = 1536
    cfg.trainer.bptt_horizon = 8
    cfg.training_env.forward_pass_minibatch_target_size = 192
    cfg.checkpointer.epoch_interval = 0

    if cfg.evaluator is not None:
        cfg.evaluator.epoch_interval = 0

    cfg.run = "smoke_test"
    return cfg


def replay() -> ReplayTool:
    env = arena.mettagrid()
    env.game.max_steps = 100
    cfg = ReplayTool(sim=SimulationConfig(suite="arena", name="eval", env=env))
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def replay_null() -> ReplayTool:
    cfg = replay()
    cfg.policy_uri = None
    return cfg


def play() -> PlayTool:
    env = arena.mettagrid()
    env.game.max_steps = 100
    cfg = PlayTool(sim=SimulationConfig(suite="arena", name="eval", env=env))
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def play_null() -> PlayTool:
    # Mock policy test.
    cfg = play()
    cfg.policy_uri = None
    return cfg
