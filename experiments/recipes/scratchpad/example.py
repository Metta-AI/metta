from typing import Sequence
from experiments.recipes import arena
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig

# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot

# You can run these functions locally with e.g. `./tools/run.py train scratchpad.{{ USER }}`
# The VSCode "Run and Debug" section supports options to run these functions.


def trainer() -> TrainerConfig:
    """Training configuration for local experimentation."""
    env = arena.env_config()
    env.game.max_steps = 100
    cfg = arena.train(
        curriculum_cfg=arena.curriculum_config(env),
    )
    assert cfg.evaluation is not None
    # When we're using this file, we training locally on code that's likely not to be checked in, let alone pushed.
    # So remote evaluation probably doesn't make sense.
    cfg.evaluation.evaluate_remote = False
    cfg.evaluation.evaluate_local = True
    return cfg


def simulation() -> SimulationConfig:
    """Simulation configuration for play/replay."""
    env = arena.sim()[0].env
    env.game.max_steps = 100
    return SimulationConfig(env=env, name="scratchpad")


def simulations() -> Sequence[SimulationConfig]:
    """Evaluation simulations."""
    return arena.sim()


# Aliases for specific tools
def play_simulation() -> SimulationConfig:
    """Simulation for play tool."""
    return simulation()


def replay_simulation() -> SimulationConfig:
    """Simulation for replay tool."""
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


def sim() -> Sequence[SimulationConfig]:
    """Alias for simulations() to support standard CLI syntax."""
    return simulations()
