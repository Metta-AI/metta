from experiments.recipes import arena
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig

# CI-friendly runtime: tiny steps and small batches.


def trainer() -> TrainerConfig:
    """CI-friendly training configuration."""
    cfg = arena.train(curriculum_cfg=arena.curriculum_config())

    # Apply CI-friendly settings
    cfg.total_timesteps = 16  # Minimal steps for smoke testing
    cfg.minibatch_size = 8
    cfg.batch_size = 1536
    cfg.bptt_horizon = 8
    cfg.forward_pass_minibatch_target_size = 192
    cfg.checkpoint.checkpoint_interval = 0
    cfg.checkpoint.wandb_checkpoint_interval = 0
    if cfg.evaluation is not None:
        cfg.evaluation.evaluate_interval = 0

    return cfg


def simulation() -> SimulationConfig:
    """CI-friendly simulation configuration for replay/play."""
    env = arena.env_config()
    env.game.max_steps = 100
    env.label = env.label or "mettagrid"
    return SimulationConfig(env=env, name="ci_test")


def replay_simulation() -> SimulationConfig:
    """Simulation config specifically for replay tool."""
    return simulation()


def replay_null() -> SimulationConfig:
    """Simulation config for replay with null policy (for testing)."""
    return simulation()


def play_simulation() -> SimulationConfig:
    """Simulation config specifically for play tool."""
    return simulation()


def play_null() -> SimulationConfig:
    """Simulation config for play with null policy (for testing)."""
    return simulation()


# Add recipe shims for standard CLI interface
def train() -> TrainerConfig:
    """Alias for trainer() to support standard CLI syntax."""
    return trainer()


def play() -> SimulationConfig:
    """Alias for play_simulation() to support standard CLI syntax."""
    return play_simulation()


def replay() -> SimulationConfig:
    """Alias for replay_simulation() to support standard CLI syntax."""
    return replay_simulation()
