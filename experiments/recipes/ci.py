import experiments.recipes
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import metta.sim.simulation_config

# CI-friendly runtime: tiny steps and small batches.


def train() -> metta.tools.train.TrainTool:
    cfg = experiments.recipes.arena.train(
        curriculum=experiments.recipes.arena.make_curriculum(
            experiments.recipes.arena.mettagrid()
        )
    )

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


def replay() -> metta.tools.replay.ReplayTool:
    env = experiments.recipes.arena.mettagrid()
    env.game.max_steps = 100
    cfg = metta.tools.replay.ReplayTool(
        sim=metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="eval", env=env
        )
    )
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    cfg.launch_viewer = False
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def replay_null() -> metta.tools.replay.ReplayTool:
    cfg = replay()
    cfg.policy_uri = None
    return cfg


def play() -> metta.tools.play.PlayTool:
    env = experiments.recipes.arena.mettagrid()
    env.game.max_steps = 100
    cfg = metta.tools.play.PlayTool(
        sim=metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="eval", env=env
        )
    )
    cfg.wandb.enabled = False
    cfg.system.vectorization = "serial"
    cfg.open_browser_on_start = False
    cfg.sim.env.label = cfg.sim.env.label or "mettagrid"
    return cfg


def play_null() -> metta.tools.play.PlayTool:
    # Mock policy test.
    cfg = play()
    cfg.policy_uri = None
    return cfg
